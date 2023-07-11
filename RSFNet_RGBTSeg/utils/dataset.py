import os
import os.path
import cv2
import numpy as np
import pickle
import torch

from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test', 'test_day', 'test_night']

    with open(os.path.join(data_root, split+'.txt'), 'r') as f:
        names = [name.strip() for name in f.readlines()]

    image_label_list = []
    # list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(names), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for name in names:
        image_name = os.path.join(data_root, 'images', name + '.png')
        label_name = os.path.join(data_root, 'labels', name + '.png')
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list

class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        with open(os.path.join(data_root, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.scores_data = None
        # get the scores for train data
        if split == 'train':
            with open(os.path.join(data_root, "score.pkl"), "rb") as fin:
                self.scores_data = pickle.load(fin)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        name  = self.names[index]
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Can read 4 channel image
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        if self.scores_data:
            # score = self.scores_data[name]['iou_rate']
            # gate1_gt = torch.zeros(1)
            # gate2_gt = torch.zeros(1)
            # gate1_gt[0] = score
            # gate2_gt[0] = 1 - score
            # return image, label, gate1_gt, gate2_gt

            rgb_iou = torch.zeros(1)
            inf_iou = torch.zeros(1)
            rgb_iou[0] = self.scores_data[name]['rgb_iou']
            inf_iou[0] = self.scores_data[name]['inf_iou']
            # print('rgb IoU: {:.3f}, thm IoU: {:.3f}'.format(rgb_iou[0], inf_iou[0]))
            return image, label, rgb_iou, inf_iou
        else:
            return image, label, name