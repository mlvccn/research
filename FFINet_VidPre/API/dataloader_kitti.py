

"""KITTI Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import hickle as hkl

logger = logging.getLogger(__name__)


class KITTIDataset(Dataset):
    def __init__(self, data_file, source_file, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='prediction', sequence_start_mode='all', N_seq=None,data_format=None):
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = 'channels_first'
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode
        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape
        self.mean = 0
        self.std = 1
        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        print(self.N_sequences)
        super(KITTIDataset,self).__init__()
    
    def __len__(self):
        return self.N_sequences

    def __getitem__(self, i):
        batch_ind = self.possible_starts[i]
        begin = batch_ind
        input = self.nt - 10
        end = batch_ind + input
        input = self.preprocess(self.X[begin:(begin+10),::])
        predict = self.preprocess(self.X[(begin+10):(end+10),:,:,:])
        return input,predict
    
    # def next(self):
    #     with self.lock:
    #         current_index = (self.batch_index * self.batch_size) % self.n
    #         index_array, current_batch_size = next(self.index_generator), self.batch_size
    #     batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
    #     for i, idx in enumerate(index_array):
    #         idx = self.possible_starts[idx]
    #         batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
    #     if self.output_mode == 'error':  # model outputs errors, so y should be zeros
    #         batch_y = np.zeros(current_batch_size, np.float32)
    #     elif self.output_mode == 'prediction':  # output actual pixels
    #         batch_y = batch_x
    #     return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255


def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):
    data_root = data_root
    dataset_train = os.path.join(data_root,'kitti/X_train.hkl')
    source_train = os.path.join(data_root, 'kitti/sources_train.hkl')
    dataset_test = os.path.join(data_root, 'kitti/X_pedest_test.hkl')
    source_test = os.path.join(data_root, 'kitti/sources_pedest_test.hkl')
    # dataset_valid = os.path.join(data_root, 'kitti/X_val.hkl')
    # source_valid = os.path.join(data_root, 'kitti/sources_val.hkl')
    train_set = KITTIDataset(data_file=dataset_train, source_file=source_train, batch_size=batch_size, shuffle=False, nt=20,sequence_start_mode='unique')
    test_set = KITTIDataset(data_file=dataset_test, source_file=source_test, batch_size=val_batch_size, shuffle=False, nt=11,sequence_start_mode='unique')
    #valid_set =KITTIDataset(data_file=dataset_valid, source_file=source_valid, batch_size=val_batch_size, shuffle=False, nt=11)
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=0)
    # dataloader_valid = torch.utils.data.DataLoader(
    #     valid_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return dataloader_train, None, dataloader_test, 0, 1

if __name__ == '__main__':

    #x = hkl.load('/home/cjj/Documents/zch_Documents/SimVP-master/data/kitti/sources_pedest_test.hkl')
    train_loader, vali_loader, test_loader, data_mean, data_std = load_data(16, 16, '/home/cjj/Documents/zch_Documents/SimVP-master/data',4)
    i = 0
    for x,y in test_loader:
        i += 1
        print(i)
        # for i in range(10):
        #     img = x[0,i,...].numpy()
        #     img = np.transpose(img, (1, 2, 0))
        #     img = (img*255).astype(np.uint8)
        #     cv2.imwrite('kitti'+str(i)+'.png',img)
        # img = y[0,...].numpy()
        # img = np.transpose(img, (1, 2, 0))
        # img = (img*255).astype(np.uint8)
        # cv2.imwrite('kitti_pred.png',img)
        #print(x.shape)
        # for i in range(16):
        #     img = y[i,0,...].numpy()
        #     img = np.transpose(img, (1, 2, 0))
        #     img = (img*255).astype(np.uint8)
        #     cv2.imwrite('kitti'+str(i)+'.png',img)
        # print(x.shape)