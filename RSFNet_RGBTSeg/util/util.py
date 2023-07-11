import numpy as np
from PIL import Image
import subprocess as sp

 
# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 
def get_palette():
    # unlabelled = [0,0,0]
    unlabelled = [200,200,200]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    error = [255, 0, 0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump, error])
    return palette

def visualize(image_name, labels, predictions, weight_name, save_file):
    palette = get_palette()
    error_num = 9
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        label = labels[i].cpu().numpy()
        h, w = pred.shape[0], pred.shape[1]
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        pred = pred.squeeze().flatten()
        label = label.squeeze().flatten()
        # print(len(pred == label))
        pred[pred != label] = error_num
        pred = np.resize(pred, (h,w))
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        # save_path = save_file +'/Pred_' + weight_name + '_' + image_name[i] + '.png'
        save_path = save_file + '/Pred_' + image_name[i] + '.png'
        img.save(save_path)
    return save_path

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_gpu_memory(index=0):
    command = "nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    current_gpu_info = memory_free_info[index].split(', ')
    str = "GPU index: {}; Name: {}; Total memory: {}; Used memory: {}".format(current_gpu_info[0], \
        current_gpu_info[1], current_gpu_info[2], current_gpu_info[3])
    return str