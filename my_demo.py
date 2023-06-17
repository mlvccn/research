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
parser.add_argument('--model_name', '-m', type=str, default='DPAdd101_addHead_NTE')
parser.add_argument('--layers', '-l', type=int, default=34)
# parser.add_argument('--weight_name', '-w', type=str, default=None) # RTFNet_152, RTFNet_50, please change the number of layers in the network file
parser.add_argument('--file_name', '-f', type=str, default='last_20220602_175347.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test') # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=1)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480) 
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=4)
parser.add_argument('--n_class', '-nc', type=int, default=9)
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

    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, 'model', args.file_name)
    print(model_file)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.') 

    result_dir = os.path.join(model_dir, 'result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))
    
    conf_total = np.zeros((args.n_class, args.n_class))
   
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
    print(model)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    model.load_state_dict(pretrained_weight['state_dict'])
    print('loaded model done!')

    batch_size = 1 # do not change this parameter!	
    mean = [ 58.6573,  65.9755,  56.4990, 100.8296]
    std = [60.2980, 59.0457, 58.0989, 47.1224]
    test_transform = transform.Compose([
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
            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            # generate confusion matrix frame-by-frame
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6,7,8]) # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            # save demo images
            visualize(image_name=names, labels=labels, predictions=logits.argmax(1), weight_name=args.weight_name, save_file=result_dir)
            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  %(args.model_name, args.weight_name, it+1, len(test_loader), names, (end_time-start_time)*1000))
 
    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
    # conf_total_matfile = os.path.join("./runs", 'conf_'+args.weight_name+'.mat')
    # conf_total_matfile = os.path.join("./runs", 'conf_'+ args.model_name + str(args.layers) + '.mat')
    # savemat(conf_total_matfile,  {'conf': conf_total}) # 'conf' is the variable name when loaded in Matlab
 
    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' %(args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu))) 
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' %(args.img_height, args.img_width)) 
    print('* the weight name: %s' %args.weight_name) 
    print('* the file name: %s' %args.file_name) 
    print("* recall per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
          %(recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3], recall_per_class[4], recall_per_class[5], recall_per_class[6], recall_per_class[7], recall_per_class[8]))
    print("* iou per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
          %(iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5], iou_per_class[6], iou_per_class[7], iou_per_class[8])) 
    print("\n* average values (np.mean(x)): \n recall: %.6f, iou: %.6f" \
          %(recall_per_class.mean(), iou_per_class.mean()))
    print("* average values (np.mean(np.nan_to_num(x))): \n recall: %.6f, iou: %.6f" \
          %(np.mean(np.nan_to_num(recall_per_class)), np.mean(np.nan_to_num(iou_per_class))))
    print('\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' %(batch_size, ave_time_cost*1000/(len(test_loader)-5), 1.0/(ave_time_cost/(len(test_loader)-5)))) # ignore the first 10 frames
    #print('\n* the total confusion matrix: ') 
    #np.set_printoptions(precision=8, threshold=np.inf, linewidth=np.inf, suppress=True)
    #print(conf_total)
    print('\n###########################################################################')
