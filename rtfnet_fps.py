# By Yuxiang Sun, Dec. 14, 2020
# Email: sun.yuxiang@outlook.com

import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import dataset, transform
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat 
from model import *

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='DPAdd161_EGCFM_k5_addHead')
parser.add_argument('--layers', '-l', type=int, default=121)
# parser.add_argument('--weight_name', '-w', type=str, default=None) # RTFNet_152, RTFNet_50, please change the number of layers in the network file
parser.add_argument('--file_name', '-f', type=str, default='last_20220602_175347.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test') # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480) # 480, 1280
parser.add_argument('--img_width', '-iw', type=int, default=640) # 640, 720
parser.add_argument('--n_class', '-nc', type=int, default=9) # 9, 5

parser.add_argument('--num_workers', '-j', type=int, default=4)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
parser.add_argument('--model_dir', '-wd', type=str, default='./runs/')
args = parser.parse_args()
#############################################################################################
 
if __name__ == '__main__':
  
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    args.weight_name = os.path.join(args.model_name, str(args.layers))

    model_dir = os.path.join(args.model_dir, args.weight_name)

    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))
   

    if args.model_name == 'MFNet' or args.model_name == 'SegNet':
        model = eval(args.model_name)(n_class=args.n_class)
    elif args.model_name == 'RTFNet':
        model = eval(args.model_name)(n_class=args.n_class, layers=args.layers)
    elif 'EGCFM' in args.model_name:
        with_gate = True; with_skip = True; early_skip = False
        model = eval(args.model_name)(n_class=args.n_class, layers=args.layers, pretrained=True, \
            with_gate=with_gate, with_skip=with_skip, early_skip=early_skip)
    else:
        model = eval(args.model_name)(n_class=args.n_class, layers=args.layers, pretrained=True)
    
    if args.gpu >= 0: model.cuda(args.gpu)

    batch_size = 1 # do not change this parameter!	
    mean = [ 58.6573,  65.9755,  56.4990, 100.8296]
    std = [60.2980, 59.0457, 58.0989, 47.1224]
    test_transform = transform.Compose([
        transform.Resize((args.img_height, args.img_width)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    )
    test_data = dataset.SemData(split=args.dataset_split, data_root=args.data_dir, transform=test_transform)
    test_loader  = DataLoader(
        dataset     = test_data,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_time = time.time()
            logits = model(images)  # logits.size(): mini_batch*num_class*480*640
            end_time = time.time()
            if it>=5: # # ignore the first 5 frames
                ave_time_cost += (end_time-start_time)
            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  %(args.model_name, args.weight_name, it+1, len(test_loader), names, (end_time-start_time)*1000))
    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' %(args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu))) 
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' %(args.img_height, args.img_width)) 
    print('* the number of classes: %d' %(args.n_class))
    print('* the weight name: %s' %args.weight_name) 
    
    print('\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' %(batch_size, ave_time_cost*1000/(len(test_loader)-5), 1.0/(ave_time_cost/(len(test_loader)-5)))) # ignore the first 10 frames
    print('\n###########################################################################')
