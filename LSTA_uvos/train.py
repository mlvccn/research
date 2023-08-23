import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import random
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms 

import config as cfg_
from dataloaders.datasets import DAVIS2017_Train, YOUTUBE_VOS_Train, TEST
import dataloaders.custom_transforms as tr
from utils.meters import AverageMeter
from utils.checkpoint import load_network_and_optimizer, load_network, save_network
from utils.learning import adjust_learning_rate, get_trainable_params
from utils.metric import pytorch_iou
from networks.nets import LSTA

from networks.layers.loss import OhemCELoss,DistillationLoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True 

def print_log(string):
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        print(string)
        

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train CFBI")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--config', type=str, default='configs.resnet101_cfbi')

    parser.add_argument('--start_gpu', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)

    parser.add_argument('--pretrained_path', type=str, default='')

    parser.add_argument('--datasets', nargs='+', type=str, default=['youtubevos'])
    parser.add_argument('--lr', type=float, default=-1.)
    parser.add_argument('--total_step', type=int, default=-1.)
    parser.add_argument('--start_step', type=int, default=-1.)

    parser.add_argument('--loss_weight', type=float, default=0.9)

    args = parser.parse_args()

    cfg = cfg_.Configuration(args.exp_name)
    
    # if args.exp_name != '':
    #     cfg.EXP_NAME = args.exp_name

    cfg.DIST_START_GPU = args.start_gpu
    if args.gpu_num > 0:
        cfg.TRAIN_GPUS = args.gpu_num
    if args.batch_size > 0:
        cfg.TRAIN_BATCH_SIZE = args.batch_size

    if args.pretrained_path != '':
        cfg.PRETRAIN_MODEL = args.pretrained_path

    if args.lr > 0:
        cfg.TRAIN_LR = args.lr
    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step
        cfg.TRAIN_START_SEQ_TRAINING_STEPS = int(args.total_step / 2)
        cfg.TRAIN_HARD_MINING_STEP = int(args.total_step / 2)
    if args.start_step > 0:
        cfg.TRAIN_START_STEP = args.start_step

    torch.distributed.init_process_group(backend="nccl")
    gpu = local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    #############################################################################
    print_log("#"*50)
    print_log('Processing datasets...')
    composed_transforms = transforms.Compose([
        tr.RandomScale(cfg.DATA_MIN_SCALE_FACTOR, cfg.DATA_MAX_SCALE_FACTOR, cfg.DATA_SHORT_EDGE_LEN),
        tr.BalancedRandomCrop(cfg.DATA_RANDOMCROP), 
        tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
        tr.Resize(cfg.DATA_RANDOMCROP),
        # tr.aug_heavy(),
        tr.ToTensor()])
    train_datasets = []
    if 'davis2017' in cfg.DATASETS:
        print_log("loading DAVIS17...")
        train_davis_dataset = DAVIS2017_Train(
            root=cfg.DIR_DAVIS, 
            full_resolution=cfg.TRAIN_DATASET_FULL_RESOLUTION,
            transform=composed_transforms, 
            repeat_time=cfg.DATA_DAVIS_REPEAT,
            curr_len=cfg.DATA_CURR_SEQ_LEN,
            rand_gap=cfg.DATA_RANDOM_GAP_DAVIS,
            rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ)
        train_datasets.append(train_davis_dataset)
    if 'youtubevos' in cfg.DATASETS:
        print_log("loading YouTube-VOS...")
        train_ytb_dataset = YOUTUBE_VOS_Train(
            root=cfg.DIR_YTB, 
            transform=composed_transforms,
            curr_len=cfg.DATA_CURR_SEQ_LEN,
            rand_gap=cfg.DATA_RANDOM_GAP_YTB,
            rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ,
            single=cfg.DATA_SINGLE)
        if cfg.DATA_SINGLE:
            print_log("using only sigle object video")
        train_datasets.append(train_ytb_dataset)
    if 'test' in cfg.DATASETS:
        print_log("loading evaluation set...")
        test_dataset = TEST(
            transform=composed_transforms,
            curr_len=cfg.DATA_CURR_SEQ_LEN)
        train_datasets.append(test_dataset)
    if len(train_datasets) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        print_log('No dataset!')
        exit(0)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    trainloader = DataLoader(
        train_dataset,
        batch_size=int(cfg.TRAIN_BATCH_SIZE / cfg.TRAIN_GPUS),
        shuffle=False,
        num_workers=cfg.DATA_WORKERS, 
        pin_memory=True, 
        sampler=train_sampler)
    print_log('Data loaded!')
    print_log("#"*50)

    ##################################################################################
    # model, optimizer, loss
    print_log('Build VOS model.')
    model = LSTA(cfg)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print_log("Let's use "+str(torch.cuda.device_count())+ " GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True
                                                        )
    if cfg.MODEL_DISTILLATION:
        from knowledgeDistillation.stm import STM
        teacher = STM().to(device)
        pretrained_teacher = torch.load(
            cfg.MODEL_DISTILLATION_PRETRAINED_TEACHER, 
            map_location=torch.device("cuda:"+str(gpu)))
        pretrained_dict = pretrained_teacher
        teacher_model_dict = teacher.state_dict()
        pretrained_dict_update = {}
        pretrained_dict_remove = []
        for k, v in pretrained_dict.items():
            prefix1= 'module.'
            prefix2 = 'module.feature_extractor.'
            if k in teacher_model_dict:
                pretrained_dict_update[k] = v
            elif k[:7] == 'module.':
                if k[7:] in teacher_model_dict:
                    pretrained_dict_update[k[7:]] = v
                else:
                    pretrained_dict_remove.append(k)
            else:
                pretrained_dict_remove.append(k)
        print_log(pretrained_dict_remove)
        teacher_model_dict.update(pretrained_dict_update)
        teacher.load_state_dict(teacher_model_dict)

        if torch.cuda.device_count() > 1:
            # print_log("Let's use "+str(torch.cuda.device_count())+ " GPUs!")
            teacher = torch.nn.parallel.DistributedDataParallel(teacher,
                                                            device_ids=[local_rank],
                                                            output_device=local_rank)
    trainable_params = get_trainable_params(
        model=model, 
        base_lr=cfg.TRAIN_LR, 
        weight_decay=cfg.TRAIN_WEIGHT_DECAY, 
        beta_wd=cfg.MODEL_GCT_BETA_WD)

    optimizer = optim.SGD(
        trainable_params, 
        lr=cfg.TRAIN_LR, 
        momentum=cfg.TRAIN_MOMENTUM, 
        nesterov=True)

    if cfg.TRAIN_TBLOG and local_rank == 0:
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(cfg.DIR_TB_LOG)

    if cfg.MODEL_DISTILLATION:
        kd_crit = DistillationLoss()

    criterion = OhemCELoss()
    ##################################################################################
    # loading pretrained model
    step = cfg.TRAIN_START_STEP
    epoch = 0
    if cfg.TRAIN_AUTO_RESUME:
        ckpts = os.listdir(cfg.DIR_CKPT)
        if len(ckpts) > 0:
            ckpts = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), ckpts))
            ckpt = np.sort(ckpts)[-1]
            cfg.TRAIN_RESUME = True
            cfg.TRAIN_RESUME_CKPT = ckpt
            cfg.TRAIN_RESUME_STEP = ckpt + 1
        else:
            cfg.TRAIN_RESUME = False
    if cfg.TRAIN_RESUME:
        resume_ckpt = os.path.join(cfg.DIR_CKPT, 'save_step_%s.pth' % (cfg.TRAIN_RESUME_CKPT))
        model, optimizer, removed_dict = load_network_and_optimizer(model, optimizer, resume_ckpt, gpu)

        if len(removed_dict) > 0:
            print_log('Remove {} from checkpoint.'.format(removed_dict))

        step = cfg.TRAIN_RESUME_STEP
        if cfg.TRAIN_TOTAL_STEPS <= step:
            print_log("Your training has finished!")
            exit()
        epoch = int(np.ceil(step / len(trainloader)))
        print_log('Resume from step {}'.format(step))
    elif cfg.PRETRAIN:
        model, removed_dict = load_network(model, cfg.PRETRAIN_MODEL, local_rank)
        print_log('Load pretrained VOS model from {}.'.format(cfg.PRETRAIN_MODEL))
        print_log('Remove {} from checkpoint.'.format(removed_dict))
    ################################################################################
    # main training

    running_losses = []
    running_ious = []
    max_itr = cfg.TRAIN_TOTAL_STEPS
    gpu =rank=local_rank
    for _ in range(cfg.DATA_CURR_SEQ_LEN):
        running_losses.append(AverageMeter())
        running_ious.append(AverageMeter())
    batch_time = AverageMeter()
    avg_obj =  AverageMeter()       
    print_log("#"*50)
    print_log('Start training.')
    model.train()

    while step < cfg.TRAIN_TOTAL_STEPS:
        train_sampler.set_epoch(epoch)
        epoch += 1
        last_time = time.time()
        for frame_idx, sample in enumerate(trainloader):
            now_lr = adjust_learning_rate(
                optimizer=optimizer, 
                base_lr=cfg.TRAIN_LR, 
                p=cfg.TRAIN_POWER, 
                itr=step, 
                max_itr=max_itr, 
                warm_up_steps=cfg.TRAIN_WARM_UP_STEPS, 
                is_cosine_decay=cfg.TRAIN_COSINE_DECAY)

            ref_imgs = sample['ref_img']  # batch_size * 3 * h * w
            prev_imgs = sample['prev_img']
            ref_labels = sample['ref_label']  # batch_size * 1 * h * w
            prev_labels = sample['prev_label']

            obj_nums = sample['meta']['obj_num']
            bs, _, h, w = ref_imgs.size()

            
            # Sequential training
            optimizer.zero_grad()

            pred_list = model(sample)

            # loss = 0
            # jump_seq_num = 2
            for idx in range(len(pred_list)):
                pred = pred_list[idx]
                if pred == None:
                    continue
                curr_labels = sample['curr_label'][idx].float().cuda(gpu)
                y = F.interpolate(pred, size=(h,w), mode='bilinear', align_corners=True)
                
                curr_labels_ = (curr_labels >= 1).squeeze(1).long() # ->[b,h,w]
                # print(y.shape)
                # print(curr_labels_.shape)
                loss = criterion(y, curr_labels_)
                if cfg.MODEL_DISTILLATION:
                    with torch.no_grad():
                        ref_labels_onehot = torch.cat([1 - prev_labels, prev_labels], dim=1).to(device)
                        # print(sample['curr_label'][idx].shape)
                        k1,v1 = teacher(prev_imgs, ref_labels_onehot, torch.tensor([1]))
                        teacher_logit = teacher(sample["curr_img"][idx],k1,v1,torch.tensor([1]))
                        prev_imgs = sample['curr_img'][idx]
                        prev_labels = sample['curr_label'][idx]

                    kd_loss = kd_crit(y, teacher_logit)

                    loss = kd_loss * (1-args.loss_weight) + loss * args.loss_weight
                obj_nums = torch.ones_like(obj_nums).cuda(gpu)
                if len(y.shape)==4:
                    if y.size(1) == 1:
                        y_ = torch.sigmoid(y)
                        pred_mask = (y_ > 0.5).int()
                    else:
                        y_ = torch.softmax(y, dim=1)
                        pred_mask = torch.argmax(y_, dim=1)
                else:
                    y_ = torch.sigmoid(y)
                # prev_labels = y_.detach()
                # pred_mask = (y_ > 0.5).int()
                # print("pred_mask.shape", pred_mask.shape)
                # print("curr_labels_.shape",curr_labels_.shape)
                iou = pytorch_iou(pred_mask, curr_labels_, obj_nums)
                loss = (loss + torch.mean(loss)) / (idx + 1)
                running_losses[idx].update(loss.item() * cfg.DATA_CURR_SEQ_LEN)
                running_ious[idx].update(iou.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN_CLIP_GRAD_NORM)
            optimizer.step()
            batch_time.update(time.time() - last_time)
            avg_obj.update(obj_nums.float().mean().item())
            last_time = time.time()

            if step % cfg.TRAIN_TBLOG_STEP == 0 and rank == 0 and cfg.TRAIN_TBLOG:
                for seq_step, running_loss, running_iou in zip(range(len(running_losses)), running_losses, running_ious):
                    tblogger.add_scalar('S{}/Loss'.format(seq_step), running_loss.avg, step)
                    tblogger.add_scalar('S{}/IoU'.format(seq_step), running_iou.avg, step)
                tblogger.add_scalar('LR', now_lr, step)
                tblogger.flush()
            if step % cfg.TRAIN_LOG_STEP == 0 and rank == 0:
                strs = 'Itr:{}, LR:{:.7f}, Time:{:.3f}, Obj:{:.1f}'.format(step, now_lr, batch_time.avg, avg_obj.avg)
                batch_time.reset()
                avg_obj.reset()
                for idx in range(cfg.DATA_CURR_SEQ_LEN):
                    strs += ', S{}: L {:.3f}({:.3f}) IoU {:.3f}({:.3f})'.format(idx, running_losses[idx].val, running_losses[idx].avg, 
                                                                                    running_ious[idx].val, running_ious[idx].avg)
                    running_losses[idx].reset()
                    running_ious[idx].reset()
                print_log(strs)
            if step % cfg.TRAIN_SAVE_STEP == 0 and step != 0 and rank == 0:
                print_log('Save CKPT (Step {}).'.format(step))
                save_network(model, optimizer, step, cfg.DIR_CKPT, cfg.TRAIN_MAX_KEEP_CKPT)
            step += 1
            if step > cfg.TRAIN_TOTAL_STEPS:
                break
    if rank == 0:
        print_log('Save final CKPT (Step {}).'.format(step - 1))
        save_network(model, optimizer, step - 1, cfg.DIR_CKPT, cfg.TRAIN_MAX_KEEP_CKPT)



if __name__ == '__main__':
    main()
#     cfg = cfg.Configuration()
#     model = LSTA(cfg)
#     print_log(model)
#     x = torch.rand(2,3,465,465)
#     y = model(x,x,x)
#     print_log(y.shape)