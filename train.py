# By Yuxiang Sun, Dec. 4, 2019
# Email: sun.yuxiang@outlook.com

import os, argparse, time, datetime, stat, shutil, logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results, AverageMeter, get_gpu_memory
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from model import *

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 
parser.add_argument('--model_name', '-m', type=str, default='RTFNet')
parser.add_argument('--layers', '-l', type=int, default=50)
#batch_size: RTFNet-152: 2; RTFNet-101: 2; RTFNet-50: 3; RTFNet-34: 10; RTFNet-18: 15;
parser.add_argument('--batch_size', '-b', type=int, default=2)
parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=300) # please stop training mannully 
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
parser.add_argument('--print_freq', type=int, default=10)
# parser.add_argument('--resume', type=str, default='./runs/MFNet0/model/last.pth')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./runs/RTFNet/model')
parser.add_argument('--save_model', type=str, default='best')
parser.add_argument('--tensorboard_dir', type=str, default='./runs/tensorboard_log')
parser.add_argument('--pretrained', default='true', type=str)
args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]

is_best = False
best_mIoU = 0.0
best_mAcc = 0.0
best_mIoU_epoch = 0
best_mAcc_epoch = 0

def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False # https://stackoverflow.com/questions/11820338/replace-default-handler-of-python-logger/11821510
    return logger

def train(epo, model, train_loader, optimizer):
    batch_time = AverageMeter()
    model.train()
    max_iter = args.epoch_max * len(train_loader)
    end = time.time()
    for it, (images, labels, names, gate_gt) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
        loss.backward()
        optimizer.step()

        # Calculate the ETA, i.e. remain time for whole training.
        batch_time.update(time.time() - end)
        end = time.time()
        current_iter = epo * len(train_loader) + it + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        lr_this_epo=0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        if (it+1) % args.print_freq == 0:
            logger.info('Train: {}, epo: {}/{}, iter: {}/{}, lr: {:.3e}, loss: {:.4f}, eta: {}' \
                .format(args.model_name, epo + 1, args.epoch_max, it+1, len(train_loader), lr_this_epo, float(loss), remain_time))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = False # note that I have not colorized the GT and predictions here
        if accIter['train'] % 500 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:,:3], nrow=8, padding=10) # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
        accIter['train'] = accIter['train'] + 1

def validation(epo, model, val_loader): 
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
            if (it+1) % args.print_freq == 0:
                logger.info('Val: {}, epo {}/{}, iter {}/{}, loss {:.4f}' \
                    .format(args.model_name, epo + 1, args.epoch_max, it + 1, len(val_loader), float(loss)))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1, 255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits.argmax(1).unsqueeze(1)*scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1

def testing(epo, model, test_loader):
    global is_best
    global best_mIoU
    global best_mAcc
    global best_mIoU_epoch
    global best_mAcc_epoch
    is_best = False
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = 'bs_' + str(args.batch_size) +'_lr_' + str(args.lr_start) + '_test_results.txt'
    testing_results_file = os.path.join(args.tensorboard_dir, testing_results_file)
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6,7,8]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            if (it+1) % args.print_freq == 0:
                logger.info('Test: {}, epoch {}/{}, iter {}/{}' .format(args.model_name, epo + 1, args.epoch_max, it+1, len(test_loader)))
    precision, recall, IoU = compute_results(conf_total)
    if IoU.mean() > best_mIoU:
        is_best = True
        best_mIoU = IoU.mean()
        best_mIoU_epoch = epo + 1
    if recall.mean() > best_mAcc:
        best_mAcc = recall.mean()
        best_mAcc_epoch = epo + 1
    writer.add_scalar('Test/average_precision',precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s"% label_list[i], recall[i],epo)
        writer.add_scalar('Test(class)/Iou_%s'% label_list[i], IoU[i], epo)
    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s\n" %(get_gpu_memory(args.gpu)))
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc=Recall %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write('\nWhole: ')
        f.write(str(epo+1)+': ')
        for i in range(len(precision)):
            f.write('(%0.4f, %0.4f), ' % (100*recall[i], 100*IoU[i]))
        f.write('(%0.4f, %0.4f)\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU))))
    if epo==args.epoch_max-1:
        with open(testing_results_file, 'a') as f:
            f.write("Last mIoU & mAcc in epoch %s are %0.4f, %0.4f\n" %(epo + 1, IoU.mean(), recall.mean()))
            f.write("Best mIoU in epoch %s is %0.4f\n" %(best_mIoU_epoch, best_mIoU))
            f.write("Best mAcc in epoch %s is %0.4f\n" %(best_mAcc_epoch, best_mAcc))
        with open(testing_results_file, "r") as file:
            writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo + 1)
    logger.info('saving testing results.')



def test_day_night(epo, model, test_loader, mode='day-time'):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    testing_results_file = 'bs_' + str(args.batch_size) +'_lr_' + str(args.lr_start) + '_test_results.txt'
    testing_results_file = os.path.join(args.tensorboard_dir, testing_results_file)
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6,7,8]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            if (it+1) % args.print_freq == 0:
                logger.info('Test: {}, epoch {}/{}, iter {}/{}' .format(args.model_name, epo + 1, args.epoch_max, it+1, len(test_loader)))
    precision, recall, IoU = compute_results(conf_total)

    with open(testing_results_file, 'a') as f:
        f.write(str(mode) + ': ')
        f.write(str(epo+1)+': ')
        for i in range(len(precision)):
            f.write('(%0.4f, %0.4f), ' % (100*recall[i], 100*IoU[i]))
        f.write('(%0.4f, %0.4f)\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU))))



if __name__ == '__main__':
    logger = get_logger()
    torch.cuda.set_device(args.gpu)
    logger.info("the pytorch version {}".format(torch.__version__))
    logger.info("the gpu count: {}".format(torch.cuda.device_count()))
    logger.info("the current used gpu: {}".format(torch.cuda.current_device()))

    if args.model_name == 'MFNet' or args.model_name == 'SegNet':
        model = eval(args.model_name)(n_class=args.n_class)
    elif args.model_name == 'SingleNet':
        if args.pretrained == 'true':
            print('pretrained: ', args.pretrained)
            model = eval(args.model_name)(n_class=args.n_class, layers=args.layers, pretrained=True)
        else:
            model = eval(args.model_name)(n_class=args.n_class, layers=args.layers, pretrained=False)
    else:
        model = eval(args.model_name)(n_class=args.n_class, layers=args.layers)

    if args.gpu >= 0: model.cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    weight_dir = os.path.join(args.save_path)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
 
    writer = SummaryWriter(args.tensorboard_dir)

    logger.info('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    logger.info('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    logger.info('weight will be saved in: %s' % weight_dir)
    logger.info(args)
    logger.info('\n{}'.format(model))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.epoch_from = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_mIoU = checkpoint['best_mIoU']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset  = MF_dataset(data_dir=args.data_dir, split='val')
    test_dataset = MF_dataset(data_dir=args.data_dir, split='test')
    test_day_dataset = MF_dataset(data_dir=args.data_dir, split='test_day')
    test_night_dataset = MF_dataset(data_dir=args.data_dir, split='test_night')

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    test_day_loader = DataLoader(
        dataset      = test_day_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    test_night_loader = DataLoader(
        dataset      = test_night_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        epoch_log = epo + 1
        logger.info('train {}, epoch # {} begin...'.format(args.model_name, epoch_log))
        #scheduler.step() # if using pytorch 0.4.1, please put this statement here 
        train(epo, model, train_loader, optimizer)
        # validation(epo, model, val_loader)
        testing(epo, model, test_loader) # testing is just for your reference, you can comment this line during training
        print("Testing day-time")
        test_day_night(epo, model, test_day_loader, 'Day-t')
        print("Testing night-time")
        test_day_night(epo, model, test_night_loader, 'Night')
        
        scheduler.step() # if using pytorch 1.1 or above, please put this statement here
        checkpoint_last = os.path.join(weight_dir, 'last_'+ args.save_model +'.pth')
        logger.info('saving check point: {} '.format(checkpoint_last))
        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), \
            'optimizer': optimizer.state_dict(), 'best_mIoU': best_mIoU}, checkpoint_last)
        if is_best:
            checkpoint_best = os.path.join(weight_dir, 'best_'+ args.save_model +'.pth')
            logger.info('saving check point: {} '.format(checkpoint_best))
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), \
                'optimizer': optimizer.state_dict(), 'best_mIoU': best_mIoU}, checkpoint_best)
            logger.info('Best mIoU in epoch {} is {:.4f}'.format(epoch_log, best_mIoU))
    logger.info('Best mIoU in epoch {} is {:.4f}'.format(best_mIoU_epoch, best_mIoU))
    logger.info('Best mAcc in epoch {} is {:.4f}'.format(best_mAcc_epoch, best_mAcc))
    
    
