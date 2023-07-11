import os, argparse, time, datetime, stat, shutil, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from utils import dataset, transform
from util.util import compute_results, AverageMeter, get_gpu_memory
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from model import *
import random

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 
parser.add_argument('--model_name', '-m', type=str, default='PAdd50_EGCFM')
parser.add_argument('--layers', '-l', type=int, default=18)
parser.add_argument('--batch_size', '-b', type=int, default=8)
parser.add_argument('--lr_start', '-ls', type=float, default=0.001)
parser.add_argument('--gpu', '-g', type=int, default=1)
#############################################################################################
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', '-wd', type=float, default=0.0005)
parser.add_argument('--index_split', type=int, default=10)
parser.add_argument('--epoch_max', '-em', type=int, default=100) # please stop training mannully 
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--aux_weight', '-aw', type=float, default=1.0)
parser.add_argument('--class_weight', type=float, default=1.02)
parser.add_argument('--data_dir', '-dr', type=str, default='./datasets/MF')
parser.add_argument('--print_freq', type=int, default=10)
# parser.add_argument('--resume', type=str, default='./batches/Add/18/lr/model/last_20211220_141747.pth')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./runs/Add/model')
parser.add_argument('--save_model', type=str, default='best')
parser.add_argument('--tensorboard_dir', type=str, default='./runs/tensorboard_log')
parser.add_argument('--pretrained', default='false', type=str)
## train for the pseudo gate fusion
parser.add_argument('--with_gate', type=str, default='false')
parser.add_argument('--with_skip', type=str, default='false')
parser.add_argument('--early_skip', type=str, default='false')
args = parser.parse_args()
#############################################################################################

is_best = False
best_mIoU = 0.0
best_mIoU_acc = 0.0
best_mAcc = 0.0
best_mIoU_epoch = 0
best_mAcc_epoch = 0
def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """ poly learning rate policy """
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def train(epo, model, train_loader, optimizer, criterion):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    model.train()
    max_iter = args.epoch_max * len(train_loader)
    end = time.time()
    for it, (images, labels, gate1_gt, gate2_gt) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        gate1_gt = Variable(gate1_gt).cuda(args.gpu)
        gate2_gt = Variable(gate2_gt).cuda(args.gpu)
        logits, gate1, gate2, aux1_logits, aux2_logits =  model(images)
        main_loss = criterion(logits, labels)
        aux1_loss = criterion(aux1_logits, labels)
        aux2_loss = criterion(aux2_logits, labels)
        gate1_loss = F.smooth_l1_loss(gate1, gate1_gt)
        gate2_loss = F.smooth_l1_loss(gate2, gate2_gt)
        loss =  main_loss + args.aux_weight * (gate1_loss + gate2_loss) + aux1_loss + aux2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        current_iter = epo * len(train_loader) + it + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        current_lr = poly_learning_rate(args.lr_start, current_iter, max_iter, power=0.9)
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
           
        if (it + 1) % args.print_freq == 0:
            logger.info('Train: {}, '
                        'Epoch: [{}/{}][{}/{}], '
                        'LR: {lr:.3e}, '
                        'ETA: {remain_time}, '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                        'Main Loss: {main_loss:.3f}, '
                        'Gate1 Loss: {gate1_loss:.3f}, '
                        'Gate2 Loss: {gate2_loss:.3f}, '
                        'Aux1 Loss:{aux1_loss:.3f}, '
                        'Aux2 Loss:{aux2_loss:.3f}, '
                        'Loss: {loss_meter.val:.4f}'.format(args.model_name,
                                                            epo + 1, args.epoch_max, it + 1, len(train_loader),
                                                            lr=current_lr,
                                                            remain_time=remain_time,
                                                            batch_time=batch_time,
                                                            main_loss=main_loss,
                                                            gate1_loss=gate1_loss,
                                                            gate2_loss=gate2_loss,
                                                            aux1_loss=aux1_loss,
                                                            aux2_loss=aux2_loss,
                                                            loss_meter=loss_meter))
        writer.add_scalar('Train/batch_loss', loss_meter.val, current_iter)
    writer.add_scalar('Train/loss', loss_meter.avg, epo)
        

def validation(epo, model, val_loader, criterion):
    loss_meter = AverageMeter()
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(images)
            loss = criterion(logits, labels)  # Note that the cross_entropy function has already include the softmax function
            loss_meter.update(loss.item(), images.size(0))
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6,7,8]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            if (it+1) % args.print_freq == 0:
                logger.info('Val: {}, Epoch: [{}/{}][{}/{}], Loss: {loss_meter.val:.4f}' \
                    .format(args.model_name, epo + 1, args.epoch_max, it + 1, len(val_loader), loss_meter=loss_meter))
            current_iter = epo * len(val_loader) + it + 1
            writer.add_scalar('Validation/batch_loss', loss_meter.val, current_iter)
            view_figure = False  # note that I have not colorized the GT and predictions here
            if current_iter % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, current_iter)
                    scale = max(1, 255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, current_iter)
                    predicted_tensor = logits.argmax(1).unsqueeze(1)*scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, current_iter)
        writer.add_scalar('Validation/loss', loss_meter.avg, epo)
    precision, recall, IoU = compute_results(conf_total)
    writer.add_scalar('Val/average_precision',precision.mean(), epo)
    writer.add_scalar('Val/average_recall', recall.mean(), epo)
    writer.add_scalar('Val/average_IoU', IoU.mean(), epo)

def testing(epo, model, test_loader):
    global is_best
    global best_mIoU
    global best_mIoU_acc
    global best_mAcc
    global best_mIoU_epoch
    global best_mAcc_epoch
    is_best = False
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = 'bs_' + str(args.batch_size) +'_lr_' + str(args.lr_start) +'_aux_' + str(args.aux_weight) + '_test_results.txt'
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
        best_mIoU_acc = recall.mean()
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
            f.write("# %s, initial lr: %s, batch size: %s, aux weight: %s, date: %s \n" %(args.model_name, args.lr_start, args.batch_size, args.aux_weight, datetime.date.today()))
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
            f.write("Best mIoU in epoch %s is %0.4f, and the mAcc is %0.4f\n" %(best_mIoU_epoch, best_mIoU, best_mIoU_acc))
            f.write("Best mAcc in epoch %s is %0.4f\n" %(best_mAcc_epoch, best_mAcc))
        with open(testing_results_file, "r") as file:
            writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo + 1)
    logger.info('saving testing results.')


def test_day_night(epo, model, test_loader, mode='day-time'):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    testing_results_file = 'bs_' + str(args.batch_size) +'_lr_' + str(args.lr_start) +'_aux_' + str(args.aux_weight) + '_test_results.txt'
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
    setup_seed(2023)
    logger = get_logger()
    torch.cuda.set_device(args.gpu)
    logger.info("the pytorch version {}".format(torch.__version__))
    logger.info("the gpu count: {}".format(torch.cuda.device_count()))
    logger.info("the current used gpu: {}".format(torch.cuda.current_device()))
    with_gate = False if args.with_gate=='false' else True
    with_skip = False if args.with_skip=='false' else True
    early_skip = False if args.early_skip=='false' else True
    print('with_gate: %s; with_skip: %s; early_skip: %s'%(with_gate, with_skip, early_skip))
    print('Thermal encoder\'s pretrained: ', args.pretrained)
    if args.pretrained == 'true':
        model = eval(args.model_name)(n_class=args.n_class, layers=args.layers, pretrained=True, \
            with_gate=with_gate, with_skip=with_skip, early_skip=early_skip)
    else:
        model = eval(args.model_name)(n_class=args.n_class, layers=args.layers, pretrained=False, \
            with_gate=with_gate, with_skip=with_skip, early_skip=early_skip)
    
    modules_pre = [] # pretrained modules
    modules_new = []
    for key, value in model._modules.items():
        if key.startswith('layer'):
            if key.endswith('rgb'):
                modules_pre.append(value)
            else:
                if args.pretrained == 'true': # if thermal pretrained is true, add its encoder layers to modules_pre. 
                    modules_pre.append(value)
                else:
                    modules_new.append(value)
        else:
            modules_new.append(value)
    args.index_split = len(modules_pre)  # the module after index_split need multiply 10 at learning rate
    print('index_split: ', args.index_split)
    params_list = []
    for module in modules_pre:
        params_list.append(dict(params=module.parameters(), lr=args.lr_start))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.lr_start * 10))
    
    if args.gpu >= 0: model.cuda(args.gpu)
    optimizer = torch.optim.SGD(params_list, lr=args.lr_start, momentum=args.momentum, weight_decay=args.weight_decay)

    weight_dir = os.path.join(args.save_path)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
        # pass
 
    writer = SummaryWriter(args.tensorboard_dir)

    logger.info('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    logger.info('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    logger.info('weight will be saved in: %s' % weight_dir)
    logger.info(args)
    logger.info('\n{}'.format(model))
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.epoch_from = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_mIoU = checkpoint['best_mIoU']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    mean = [ 58.6573,  65.9755,  56.4990, 100.8296]
    std = [60.2980, 59.0457, 58.0989, 47.1224]
    scale_min, scale_max = 0.5, 2.0
    rotate_min, rotate_max = -10, 10
    train_h = 256  # default: 480
    train_w = 512  # default: 640
    train_transform = transform.Compose([
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([train_h, train_w], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    train_data = dataset.SemData(split='train', data_root=args.data_dir, transform=train_transform)

    val_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    )
    val_data = dataset.SemData(split='val', data_root=args.data_dir, transform=val_transform)
    test_data = dataset.SemData(split='test', data_root=args.data_dir, transform=val_transform)
    test_day = dataset.SemData(split='test_day', data_root=args.data_dir, transform=val_transform)
    test_night = dataset.SemData(split='test_night', data_root=args.data_dir, transform=val_transform)

    train_loader  = DataLoader(
        dataset     = train_data,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        shuffle     = True,
        pin_memory  = True,
        drop_last   = True
    )
    val_loader  = DataLoader(
        dataset     = val_data,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        shuffle     = False,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_data,
        batch_size   = args.batch_size,
        num_workers = args.num_workers,
        shuffle      = False,
        pin_memory   = True,
        drop_last    = False
    )
    test_day_loader = DataLoader(
        dataset      = test_day,
        batch_size   = args.batch_size,
        num_workers = args.num_workers,
        shuffle      = False,
        pin_memory   = True,
        drop_last    = False
    )
    test_night_loader = DataLoader(
        dataset      = test_night,
        batch_size   = args.batch_size,
        num_workers = args.num_workers,
        shuffle      = False,
        pin_memory   = True,
        drop_last    = False
    )
    # criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(args.gpu)
    class_weights = None
    from utils.calculate_weights import calculate_weigths_labels
    class_weights = calculate_weigths_labels('mf', train_loader, args.n_class, c=args.class_weight)
    class_weights = torch.FloatTensor(class_weights).cuda(args.gpu)
    print('class weights: ', class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255).cuda(args.gpu)
    for epo in range(args.epoch_from, args.epoch_max):
        epoch_log = epo + 1
        logger.info('train {}, epoch # {} begin...'.format(args.model_name, epoch_log))
        train(epo, model, train_loader, optimizer, criterion)
        validation(epo, model, val_loader, criterion)
        testing(epo, model, test_loader) # testing is just for your reference, you can comment this line during training
        print("Testing day-time")
        test_day_night(epo, model, test_day_loader, 'Day-t')
        print("Testing night-time")
        test_day_night(epo, model, test_night_loader, 'Night')
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
