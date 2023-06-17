# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""KTH Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from API.utils import create_random_shape_with_random_motion
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

def get_binary_mask(x):
    error = torch.abs(x-x[:,0:1]).sum(dim=1)
    binary_mask = error<0.05
    y = binary_mask[0]
    for t in range(1,binary_mask.shape[0]):
        y = y&binary_mask[t]
    return y

class KTHDataset(Dataset):
    def __init__(self, train, datas, indices, pre_seq_length, aft_seq_length, require_back=False):
        super(KTHDataset,self).__init__()
        self.datas = datas.swapaxes(2, 3).swapaxes(1,2)
        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.require_back = require_back
        self.mean = 0
        self.std = 1
        self.train = train
        self.mask =  np.load('/data16t/KTH/kth/kth_mask.npy')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = begin + self.pre_seq_length + self.aft_seq_length
        data = torch.tensor(self.datas[begin:end1,::]).float()
        labels = torch.tensor(self.datas[end1:end2,::]).float()
        if self.train:
          all_masks = create_random_shape_with_random_motion(
            self.pre_seq_length, imageHeight=128, imageWidth=128)
          all_masks = np.stack([np.expand_dims(x, 2) for x in all_masks], axis=0)/ 255.0  
        else:
          all_masks = self.mask[:,i]
        if self.require_back:
            b_mask = get_binary_mask(data)
            H,W = b_mask.shape
            return data, labels, b_mask.view(1, 1, H, W)
        else:
            return data, labels, torch.tensor(all_masks.transpose(0,3,1,2),dtype=torch.float)


class InputHandle(object):
  """Class for handling dataset inputs."""

  def __init__(self, datas, indices, input_param):
    self.name = input_param['name']
    self.input_data_type = input_param.get('input_data_type', 'float32')
    self.minibatch_size = input_param['minibatch_size']
    self.image_width = input_param['image_width']
    self.datas = datas
    self.indices = indices
    self.current_position = 0
    self.current_batch_indices = []
    self.current_input_length = input_param['seq_length']

  def total(self):
    return len(self.indices)

  def begin(self, do_shuffle=True):
    logger.info('Initialization for read data ')
    if do_shuffle:
      random.shuffle(self.indices)
    self.current_position = 0
    self.current_batch_indices = self.indices[self.current_position:self
                                              .current_position +
                                              self.minibatch_size]

  def next(self):
    self.current_position += self.minibatch_size
    if self.no_batch_left():
      return None
    self.current_batch_indices = self.indices[self.current_position:self
                                              .current_position +
                                              self.minibatch_size]

  def no_batch_left(self):
    if self.current_position + self.minibatch_size >= self.total():
      return True
    else:
      return False

  def get_batch(self):
    """Gets a mini-batch."""
    if self.no_batch_left():
      logger.error(
          'There is no batch left in %s.'
          'Use iterators.begin() to rescan from the beginning.',
          self.name)
      return None
    input_batch = np.zeros(
        (self.minibatch_size, self.current_input_length, self.image_width,
         self.image_width, 1)).astype(self.input_data_type)
    for i in range(self.minibatch_size):
      batch_ind = self.current_batch_indices[i]
      begin = batch_ind
      end = begin + self.current_input_length
      data_slice = self.datas[begin:end, :, :, :]
      input_batch[i, :self.current_input_length, :, :, :] = data_slice
      # logger.info('data_slice shape')
      # logger.info(data_slice.shape)
      # logger.info(input_batch.shape)
    input_batch = input_batch.astype(self.input_data_type)
    return input_batch

  def print_stat(self):
    logger.info('Iterator Name: %s', self.name)
    logger.info('    current_position: %s', str(self.current_position))
    logger.info('    Minibatch Size %s: ', str(self.minibatch_size))
    logger.info('    total Size: %s', str(self.total()))
    logger.info('    current_input_length: %s', str(self.current_input_length))
    logger.info('    Input Data Type: %s', str(self.input_data_type))


class DataProcess(object):
  """Class for preprocessing dataset inputs."""

  def __init__(self, input_param):
    self.paths = input_param['paths']
    self.category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
    self.category_2 = ['jogging', 'running']
    self.category = self.category_1 + self.category_2
    self.image_width = input_param['image_width']

    # self.train_person = ['01']
    # self.test_person = ['17']

    self.train_person = [
        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
        '13', '14', '15', '16'
    ]
    self.test_person = ['17', '18', '19', '20', '21', '22', '23', '24', '25']

    self.input_param = input_param
    self.seq_len = input_param['seq_length']

  def load_data(self, path, mode='train'):
    """Loads the dataset.

    Args:
      path: action_path.
      mode: Training or testing.

    Returns:
      A dataset and indices of the sequence.
    """
    # path = paths[0]
    if mode == 'train':
      person_id = self.train_person
    elif mode == 'test':
      person_id = self.test_person
    else:
      print('ERROR!')
    print('begin load data' + str(path))

    frames_np = []
    # if mode != 'test':
    # TEST_CNT = 0
    frames_file_name = []
    frames_person_mark = []
    frames_category = []
    person_mark = 0

    c_dir_list = self.category
    frame_category_flag = -1
    for c_dir in c_dir_list:  # handwaving
      if c_dir in self.category_1:
        frame_category_flag = 1  # 20 step
      elif c_dir in self.category_2:
        frame_category_flag = 2  # 3 step
      else:
        print('category error!!!')

      c_dir_path = os.path.join(path, c_dir)
      p_c_dir_list = os.listdir(c_dir_path)
      # p_c_dir_list.sort() # for date seq

      for p_c_dir in p_c_dir_list:  # person01_handwaving_d1_uncomp
        # print(p_c_dir)
        if p_c_dir[6:8] not in person_id:
          continue
        person_mark += 1

        dir_path = os.path.join(c_dir_path, p_c_dir)
        filelist = os.listdir(dir_path)
        filelist.sort()  # tocheck
        for cur_file in filelist:  # image_0257
          if not cur_file.startswith('image'):
            continue

          frame_im = Image.open(os.path.join(dir_path, cur_file))
          frame_np = np.array(frame_im)  # (1000, 1000) numpy array
          # print(frame_np.shape)
          frame_np = frame_np[:, :, 0]  #
          frames_np.append(frame_np)
          frames_file_name.append(cur_file)
          frames_person_mark.append(person_mark)
          frames_category.append(frame_category_flag)

          # if mode != 'test':
          # TEST_CNT += 1
          # if TEST_CNT > 100:
          #   break


    # is it a begin index of sequence
    indices = []
    index = len(frames_person_mark) - 1
    while index >= self.seq_len - 1:
      if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
        end = int(frames_file_name[index][6:10])
        start = int(frames_file_name[index - self.seq_len + 1][6:10])
        # TODO(yunbo): mode == 'test'
        if end - start == self.seq_len - 1:
          indices.append(index - self.seq_len + 1)
          if frames_category[index] == 1:
            index -= self.seq_len - 1
          elif frames_category[index] == 2:
            index -= 2
          else:
            print('category error 2 !!!')
      index -= 1

    frames_np = np.asarray(frames_np)
    data = np.zeros((frames_np.shape[0], self.image_width, self.image_width, 1))
    for i in range(len(frames_np)):
      temp = np.float32(frames_np[i, :, :])
      data[i, :, :, 0] = cv2.resize(temp,
                                    (self.image_width, self.image_width)) / 255
    print('there are ' + str(data.shape[0]) + ' pictures')
    print('there are ' + str(len(indices)) + ' sequences')
    return data, indices

  def get_train_input_handle(self):
    if not os.path.exists(self.paths+'kth/'+'train_data_gzip.hkl'):
      train_data, train_indices = self.load_data(self.paths, mode='train')
      hkl.dump(train_data, self.paths+'train_data_gzip.hkl', mode='w',compression='gzip')
      hkl.dump(train_indices, self.paths+'train_indices_gzip.hkl', mode='w', compression='gzip')
    else:
      train_data = hkl.load(self.paths+'kth/'+'train_data_gzip.hkl')
      train_indices = hkl.load(self.paths+'kth/'+'train_indices_gzip.hkl')
    return InputHandle(train_data, train_indices, self.input_param)

  def get_test_input_handle(self):
    if not os.path.exists(self.paths+'kth/'+'test_data_gzip.hkl'):
      test_data, test_indices = self.load_data(self.paths, mode='test')
      hkl.dump(test_data, self.paths+'kth/'+'test_data_gzip.hkl', mode='w', compression='gzip')
      hkl.dump(test_indices, self.paths+'kth/'+'test_indices_gzip.hkl', mode='w', compression='gzip')
    else:
      test_data = hkl.load(self.paths+'kth/'+'test_data_gzip.hkl')
      test_indices = hkl.load(self.paths+'kth/'+'test_indices_gzip.hkl')
    return InputHandle(test_data, test_indices, self.input_param)

def load_data(batch_size, val_batch_size, data_root, pre_seq_length=10, aft_seq_length=20, require_back=False):
    img_width = 128
    pre_seq_length, aft_seq_length = 10, 20
    input_param = {
        'paths': data_root,
        'image_width': img_width,
        'minibatch_size': batch_size,
        'seq_length': (pre_seq_length + aft_seq_length),
        'input_data_type': 'float32',
        'name': 'kth'
    }
    input_handle = DataProcess(input_param)
    train_input_handle = input_handle.get_train_input_handle()
    test_input_handle = input_handle.get_test_input_handle()

    train_set = KTHDataset( 1,train_input_handle.datas,
                              train_input_handle.indices,
                              pre_seq_length,
                              pre_seq_length,
                              require_back=require_back)
    test_set = KTHDataset( 0,test_input_handle.datas,
                              test_input_handle.indices,
                              pre_seq_length,
                              aft_seq_length,
                              require_back=require_back)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=1)
    dataloader_validation = None
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=False, num_workers=1)

    return dataloader_train, dataloader_validation, dataloader_test, 0, 1


if __name__ == '__main__':
  mask =  np.load('/data16t/KTH/kth/kth_mask.npy')
  a = mask[:,0]
  print('ok')
  train_loader, vali_loader, test_loader, data_mean, data_std = load_data(16, 16, '/data16t/KTH/', 10, 20)
  print(len(test_loader))
  for x,y in test_loader:
    print(x.shape)