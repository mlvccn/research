import sys
sys.path.append('.')
sys.path.append('..')
# from networks.engine.eval_manager import Evaluator
import importlib
import os
import time
import datetime as datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms 
import numpy as np
from dataloaders.datasets import YOUTUBE_VOS_Test, DAVIS_Test, EVAL_TEST,YTO_Test,ViSal_Test, FBMS_Test
import dataloaders.custom_transforms as tr


from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder
from networks.nets import LSTA

THRESHOLD = 0.5

class Evaluator(object):
    def __init__(self, cfg):
        self.gpu = cfg.TEST_GPU_ID
        self.cfg = cfg
        self.print_log(cfg.__dict__)
        print("Use GPU {} for evaluating".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        
        self.print_log('Build VOS model.')

        self.model = LSTA(cfg).cuda(self.gpu)

        self.process_pretrained_model()

        self.prepare_dataset()

    def process_pretrained_model(self):
        cfg = self.cfg
        if cfg.TEST_CKPT_PATH == 'test':
            self.ckpt = 'test'
            self.print_log('Test evaluation.')
            return
        if cfg.TEST_CKPT_PATH is None:
            if cfg.TEST_CKPT_STEP is not None:
                ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                ckpts = os.listdir(cfg.DIR_CKPT)
                if len(ckpts) > 0:
                    ckpts = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), ckpts))
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log('No checkpoint in {}.'.format(cfg.DIR_CKPT))
                    exit()
            self.ckpt = ckpt
            cfg.TEST_CKPT_PATH = os.path.join(cfg.DIR_CKPT, 'save_step_%s.pth' % ckpt)
            self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.gpu)
            if len(removed_dict) > 0:
                self.print_log('Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load latest checkpoint from {}'.format(cfg.TEST_CKPT_PATH))
        else:
            self.ckpt = 'unknown'
            self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.gpu)
            if len(removed_dict) > 0:
                self.print_log('Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load checkpoint from {}'.format(cfg.TEST_CKPT_PATH))

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE, cfg.TEST_FLIP, cfg.TEST_MULTISCALE), 
            tr.MultiToTensor()])
        
        eval_name = '{}_{}_ckpt_{}'.format(cfg.TEST_DATASET, cfg.EXP_NAME, self.ckpt)
        if cfg.TEST_FLIP:
            eval_name += '_flip'
        if len(cfg.TEST_MULTISCALE) > 1:
            eval_name += '_ms'

        if cfg.TEST_DATASET == 'youtubevos':
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            self.dataset = YOUTUBE_VOS_Test(
                root=cfg.DIR_YTB_EVAL, 
                transform=eval_transforms,  
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2017':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=cfg.TEST_DATASET_SPLIT, 
                root=cfg.DIR_DAVIS, 
                year=2017, 
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION, 
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2016':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=cfg.TEST_DATASET_SPLIT, 
                root=cfg.DIR_DAVIS, 
                year=2016, 
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION, 
                result_root=self.result_root)
        elif cfg.TEST_DATASET == 'yto':
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            #root='./YTO',transform=None, rgb=False, result_root=None
            print("loading youtube objects.")
            self.dataset = YTO_Test(
                root=cfg.DIR_YTO_EVAL, 
                transform=eval_transforms,
                result_root=self.result_root)
        elif cfg.TEST_DATASET == "ViSal":
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            #root='./YTO',transform=None, rgb=False, result_root=None
            print("loading visal data.")
            self.dataset = ViSal_Test(
                root=cfg.DIR_VISAL_EVAL, 
                transform=eval_transforms,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == "FBMS":
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            #root='./YTO',transform=None, rgb=False, result_root=None
            print("loading FBMS data.")
            self.dataset = FBMS_Test(
                root=cfg.DIR_FBMS_EVAL, 
                transform=eval_transforms,
                result_root=self.result_root)
        elif cfg.TEST_DATASET == 'test':
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            self.dataset = EVAL_TEST(eval_transforms, self.result_root)
        else:
            print('Unknown dataset!')
            exit()

        print('Eval {} on {}:'.format(cfg.EXP_NAME, cfg.TEST_DATASET))
        self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
        self.zip_dir = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, '{}.zip'.format(eval_name))
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root)
        self.print_log('Done!')

    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        video_num = 0 
        total_time = 0
        total_frame = 0
        total_sfps = 0
        total_video_num = len(self.dataset)
        for seq_idx, seq_dataset in enumerate(self.dataset):
            video_num += 1
            seq_name = seq_dataset.seq_name
            print('Prcessing Seq {} [{}/{}]:'.format(seq_name, video_num, total_video_num))
            torch.cuda.empty_cache()

            seq_dataloader=DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=cfg.TEST_WORKERS, pin_memory=True)
            
            seq_total_time = 0
            seq_total_frame = 0
            ref_masks = []
            prev_mask = []
            with torch.no_grad():
                # previous_label = None
                feat_list = []
                frame_names = []
                first_N = False
                if cfg.MODEL_MAX_MEM > len(seq_dataloader):
                    for frame_idx, samples in enumerate(seq_dataloader):
                        time_start = time.time()
                        all_preds = []
                        if frame_idx == 0:
                            obj_num = samples[0]['meta']['obj_num']
                            imgname = samples[0]['meta']['current_name']
                            ori_height = samples[0]['meta']['height']
                            ori_width = samples[0]['meta']['width']
                            pred_label = samples[0]['current_label']
                            pred_label = pred_label.squeeze()
                            one_frametime = time.time() - time_start
                            seq_total_time += one_frametime
                            seq_total_frame += 1
                            obj_num = obj_num[0].item()
                            print('Frame: {}, Obj Num: {}, Time: {}'.format(imgname[0], obj_num, one_frametime))
                            save_mask(pred_label, os.path.join(self.result_root, seq_name, imgname[0].split('.')[0]+'.png'))
                        # fisrt N-1 frames feature extraction
                        mem_features = self.model.forward_mem(samples[0]['current_img'].cuda(self.gpu))
                        feat_list.append(mem_features)
                        if frame_idx == 0:
                            pass
                        else:
                            frame_names.append(samples[0]['meta']['current_name'][0])
                    if len(feat_list) < cfg.MODEL_MAX_MEM:
                        for _ in range(cfg.MODEL_MAX_MEM -len(feat_list)):
                            feat_list.append(feat_list[-1])
                    for tmp_idx in range(1,len(seq_dataloader)):
                        all_pred = self.model.forward_curr(feat_list,tmp_idx)
                        # import pdb; pdb.set_trace()
                        h,w = samples[0]['current_img'].shape[-2:]
                        all_preds.append(all_pred)

                        pred_label = F.interpolate(all_pred,size=(ori_height,ori_width),mode='bilinear', align_corners=True)
                        pred_label = torch.argmax(torch.softmax(pred_label, dim=1), dim=1).long()
                        # pred_label = (torch.sigmoid(pred_label) >THRESHOLD).long()
                        pred_label = pred_label.squeeze()
                        one_frametime = time.time() - time_start
                        seq_total_time += one_frametime
                        seq_total_frame += 1
                        obj_num_ = 1
                        # import pdb; pdb.set_trace()
                        print('Frame: {}, Obj Num: {}, Time: {}'.format(frame_names[tmp_idx-1], obj_num_, one_frametime))
                        save_mask(pred_label, os.path.join(self.result_root, seq_name, frame_names[tmp_idx-1].split('.')[0]+'.png'))
                else:
                    for frame_idx, samples in enumerate(seq_dataloader):
                        time_start = time.time()
                        all_preds = []
                        if frame_idx+1 < cfg.MODEL_MAX_MEM:
                            if frame_idx == 0:
                                obj_num = samples[0]['meta']['obj_num']
                                imgname = samples[0]['meta']['current_name']
                                ori_height = samples[0]['meta']['height']
                                ori_width = samples[0]['meta']['width']
                                pred_label = samples[0]['current_label']
                                pred_label = pred_label.squeeze()
                                one_frametime = time.time() - time_start
                                seq_total_time += one_frametime
                                seq_total_frame += 1
                                obj_num = obj_num[0].item()
                                print('Frame: {}, Obj Num: {}, Time: {}'.format(imgname[0], obj_num, one_frametime))
                                
                                save_mask(pred_label, os.path.join(self.result_root, seq_name, imgname[0].split('.')[0]+'.png'))
                            # fisrt N-1 frames feature extraction
                            mem_features = self.model.forward_mem(samples[0]['current_img'].cuda(self.gpu))
                            feat_list.append(mem_features)
                            if frame_idx == 0:
                                pass
                            else:
                                frame_names.append(samples[0]['meta']['current_name'][0])
                            continue

                        elif frame_idx+1 == cfg.MODEL_MAX_MEM:
                            # frame 1 to frame N segmentation
                            first_N = True
                            mem_features = self.model.forward_mem(samples[0]['current_img'].cuda(self.gpu))
                            feat_list.append(mem_features)
                            frame_names.append(samples[0]['meta']['current_name'][0])
                            # import pdb; pdb.set_trace()
                            for tmp_idx in range(1,cfg.MODEL_MAX_MEM):
                                # import pdb; pdb.set_trace()
                                all_pred = self.model.forward_curr(feat_list,tmp_idx)
                                h,w = samples[0]['current_img'].shape[-2:]
                                all_preds.append(all_pred)
                            del feat_list[0]
                        else:
                            # frame N+1 to last frame
                            first_N = False
                            mem_features = self.model.forward_mem(samples[0]['current_img'].cuda(self.gpu))
                            feat_list.append(mem_features)
                            all_pred = self.model.forward_curr(feat_list,-1)
                            h,w = samples[0]['current_img'].shape[-2:]
                            all_preds.append(all_pred)
                            del feat_list[0]
                        if len(samples) == 1: #single scale
                            obj_num = samples[0]['meta']['obj_num']
                            imgname = samples[0]['meta']['current_name']
                            ori_height = samples[0]['meta']['height']
                            ori_width = samples[0]['meta']['width']

                            if first_N:
                                # import pdb; pdb.set_trace()
                                for j in range(len(all_preds)):
                                    pred_label = F.interpolate(all_preds[j],size=(ori_height,ori_width),mode='bilinear', align_corners=True)
                                    pred_label = torch.argmax(torch.softmax(pred_label, dim=1), dim=1).long()
                                    # pred_label = (torch.sigmoid(pred_label) >THRESHOLD).long()
                                    pred_label = pred_label.squeeze()
                                    one_frametime = (time.time() - time_start)/len(all_preds)
                                    seq_total_time += one_frametime
                                    seq_total_frame += 1
                                    obj_num_ = obj_num[0].item()
                                    print('Frame: {}, Obj Num: {}, Time: {}'.format(frame_names[j], obj_num_, one_frametime))
                                    save_mask(pred_label, os.path.join(self.result_root, seq_name, frame_names[j].split('.')[0]+'.png'))
                            else:
                                pred_label = F.interpolate(all_preds[0],size=(ori_height,ori_width),mode='bilinear', align_corners=True)
                                pred_label = torch.argmax(torch.softmax(pred_label, dim=1), dim=1).long()
                                # pred_label = (torch.sigmoid(pred_label) >THRESHOLD).long()
                                # pred_label = (pred_label >0.5).long()
                                pred_label = pred_label.squeeze()
                                one_frametime = time.time() - time_start
                                seq_total_time += one_frametime
                                seq_total_frame += 1
                                obj_num = obj_num[0].item()
                                print('Frame: {}, Obj Num: {}, Time: {}'.format(imgname[0], obj_num, one_frametime))
                                # Save result
                                save_mask(pred_label, os.path.join(self.result_root, seq_name, imgname[0].split('.')[0]+'.png'))
                        else:# multi-scale
                            raise NotImplementedError
            seq_avg_time_per_frame = seq_total_time / seq_total_frame
            total_time += seq_total_time
            total_frame += seq_total_frame
            total_avg_time_per_frame = total_time / total_frame
            total_sfps += seq_avg_time_per_frame
            avg_sfps = total_sfps / (seq_idx + 1)
            print("Seq {} FPS: {}, Total FPS: {}, FPS per Seq: {}".format(seq_name, 1./seq_avg_time_per_frame, 1./total_avg_time_per_frame, 1./avg_sfps))

        # zip_folder(self.source_folder, self.zip_dir)
        self.print_log('Save result to {}.'.format(self.zip_dir))
        

    def print_log(self, string):
        print(string)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eval LST")
    parser.add_argument('--exp_name', type=str, default='')

    parser.add_argument('--config', type=str, default='config')
    
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--ckpt_step', type=int, default=-1)

    parser.add_argument('--dataset', type=str, default='')

    parser.add_argument('--flip', action='store_true')
    parser.set_defaults(flip=False)
    parser.add_argument('--ms', nargs='+', type=float, default=[1.])
    parser.add_argument('--max_long_edge', type=int, default=-1)

    parser.add_argument('--float16', action='store_true')
    parser.set_defaults(float16=False)
    parser.add_argument('--global_atrous_rate', type=int, default=1)
    parser.add_argument('--global_chunks', type=int, default=4)
    parser.add_argument('--no_local_parallel', dest='local_parallel', action='store_false')
    parser.set_defaults(local_parallel=True)
    args = parser.parse_args()

    config = importlib.import_module(args.config)
    cfg = config.Configuration(args.exp_name)
    
    cfg.TEST_GPU_ID = args.gpu_id
    # if args.exp_name != '':
    #     cfg.EXP_NAME = args.exp_name

    if args.ckpt_path != '':
        cfg.TEST_CKPT_PATH = args.ckpt_path
    if args.ckpt_step > 0:
        cfg.TEST_CKPT_STEP = args.ckpt_step

    if args.dataset != '':
        cfg.TEST_DATASET = args.dataset

    cfg.TEST_FLIP = args.flip
    cfg.TEST_MULTISCALE = args.ms
    if args.max_long_edge > 0:
        cfg.TEST_MAX_SIZE = args.max_long_edge
    else:
        cfg.TEST_MAX_SIZE = 800 * 1.3 if cfg.TEST_MULTISCALE == [1.] else 800

    cfg.MODEL_FLOAT16_MATCHING = args.float16
    cfg.TEST_GLOBAL_ATROUS_RATE = args.global_atrous_rate
    cfg.TEST_GLOBAL_CHUNKS = args.global_chunks
    cfg.TEST_LOCAL_PARALLEL = args.local_parallel

    evaluator = Evaluator(cfg=cfg)
    evaluator.evaluating()

if __name__ == '__main__':
    main()