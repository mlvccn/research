import os
from urllib.request import proxy_bypass
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils import dataset, transform

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/raid/liqiufu/DATA/VOC'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/raid/liqiufu/DATA/Cityscapes'     # foler that contains leftImg8bit/
        elif dataset.lower() == 'coco':
            return '/raid/liqiufu/DATA/COCO/2017'
        elif dataset.lower() == 'mf':
            return '/data12t/MF_dataset/dataset'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


def calculate_weigths_labels(dataset, dataloader, num_classes, c=1.02):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample[1]
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(c + (frequency / total_frequency)))
        class_weights.append(class_weight)
    class_weights = [float('{:.4f}'.format(i)) for i in class_weights]
    # ret = np.array(class_weights)
    # classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    # np.save(classes_weights_path, ret)

    return class_weights


def calculate_with_probability(c=1.02, probabilities=None):
    class_weights = []
    for probability in probabilities:
        class_weight = 1 / (np.log(c + probability)) # formulation 1, reference to ENet
        class_weights.append(class_weight)
    return class_weights


if __name__ == '__main__':
    print(Path.db_root_dir('mf'))
    data_dir='./dataset'
    n_class=9
    mean = [ 58.6573,  65.9755,  56.4990, 100.8296]
    std = [60.2980, 59.0457, 58.0989, 47.1224]
    scale_min, scale_max = 0.5, 2.0
    rotate_min, rotate_max = -10, 10
    train_h = 256  # default: 480
    train_w = 512  # default: 640
    train_transform = transform.Compose([
        transform.RandScale([scale_min, scale_max]),
        transform.RandRotate([rotate_min, rotate_max], padding=mean, ignore_label=255),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([train_h, train_w], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    val_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    )

    train_data = dataset.SemData(split='train', data_root=data_dir, transform=train_transform)
    val_data = dataset.SemData(split='val', data_root=data_dir, transform=val_transform)
    test_data = dataset.SemData(split='test', data_root=data_dir, transform=val_transform)

    train_loader  = DataLoader(
        dataset     = train_data,
        batch_size  = 1,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = False
    )
    val_loader  = DataLoader(
        dataset     = val_data,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_data,
        batch_size   = 1,
        shuffle      = False,
        num_workers  = 4,
        pin_memory   = True,
        drop_last    = False
    )
    # train_weights = calculate_weigths_labels('mf', train_loader, n_class, c=1.02)
    # val_weights = calculate_weigths_labels('mf', val_loader, n_class)
    # test_weights = calculate_weigths_labels('mf', test_loader, n_class)
    # print('train weight for MF dataset:', train_weights)
    # print('val weight for MF dataset:', val_weights)
    # print('test weight for MF dataset:', test_weights)
    train_weights_wo_argument = [1.5106, 15.8163, 30.0805, 41.7055, 38.3325, 39.7838, 46.9887, 46.4686, 44.2374]
    train_weights_w_argument = [1.5417, 12.8788, 25.1029, 38.803, 37.8386, 38.7653, 45.7684, 45.8194, 44.9238] # each time will be different
    val_weights = [1.5087, 18.6639, 34.7625, 27.0758, 39.4276, 41.9127, 48.9821, 43.9793, 41.4888]
    test_weights =  [1.4998, 17.1037, 32.6016, 33.3387, 40.5116, 44.0843, 50.1326, 48.7933, 46.9751]
    # unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump

    probabilities = [92.138, 4.145, 1.198, 0.915, 0.588, 0.449, 0.095, 0.18, 0.291] # reference to MFNet paper
    for i in range(len(probabilities)):
        probabilities[i] /= 100
    class_weights = calculate_with_probability(c=1.04, probabilities=probabilities)
    class_weights = [float('{:.4f}'.format(i)) for i in class_weights]
    print('calculate_with_probability: ', class_weights)
    class_weigths_c102 = [1.5074, 16.7684, 31.7669, 34.8029, 39.1377, 41.331, 48.231, 46.3698, 44.1472]
    class_weigths_c103 = [1.4958, 14.4901, 24.3174, 26.0396, 28.3677, 29.4911, 32.8076, 31.9439, 30.8832]
    class_weigths_c104 = [1.4845, 12.7709, 19.7339, 20.8419, 22.2923, 22.9733, 24.9167, 24.42, 23.8011]
    class_weigths_c105 = [1.4733, 11.4276, 16.6292, 17.4014, 18.391, 18.8476, 20.1229, 19.8008, 19.3957]
    class_weigths_c110 = [1.4209, 7.5586, 9.4213, 9.6531, 9.9363, 10.062, 10.3979, 10.3151, 10.2091]
    class_weigths_c115 = [1.3732, 5.7087, 6.6611, 6.7711, 6.9031, 6.9609, 7.113, 7.0758, 7.0279]
    class_weigths_c120 = [1.3297, 4.6236, 5.2014, 5.2654, 5.3416, 5.3747, 5.4611, 5.4401, 5.4129]
    class_weigths_c150 = [1.1308, 2.3109, 2.4188, 2.4299, 2.4427, 2.4483, 2.4625, 2.459, 2.4546]
    class_weigths_c200 = [0.9328, 1.4012, 1.4304, 1.4333, 1.4366, 1.438, 1.4417, 1.4408, 1.4397]
    class_weigths_c600 = [0.5169, 0.556, 0.5575, 0.5576, 0.5578, 0.5579, 0.5581, 0.558, 0.558]