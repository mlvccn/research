from API.utils import create_random_shape_with_random_motion
import os
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import torch


class Human_Occulusion_Train(Dataset):
    def __init__(self, data_dir, nt_cond, nt_pred, mask):
        super(Human_Occulusion_Train,self).__init__()
        self.data_dir = data_dir
        self.pred_h = nt_pred
        self.lb = nt_cond
        self.image_height = 128
        self.image_width = 128
        self.seq_len = nt_cond + nt_pred
        self.data, self.indeces = self.load_data(data_dir)
        self.mean = 0
        self.std = 1
        self.with_mask = mask
        # self._to_tensors = transforms.Compose([
        #     Stack(),
        #     #ToTorchFormatTensor(),
        # ])
    def load_data(self, paths):
        # data_dir = paths
        # intervel = 2

        # frames_np = []
        # scenarios = ['Walking']
        # subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        # #subjects = ['S1']
        # _path = data_dir
        # print('Using dataset Human3.6M(Train) now')
        # #print ('load data...', _path)
        # filenames = os.listdir(_path)
        # filenames.sort()
        # print ('data size ', len(filenames))
        # frames_file_name = []
        # for filename in filenames:
        #     fix = filename.split('.')
        #     fix = fix[0]
        #     subject = fix.split('_')
        #     scenario = subject[1]
        #     subject = subject[0]
        #     if subject not in subjects or scenario not in scenarios:
        #         continue
        #     file_path = os.path.join(_path, filename)
        #     #image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        #     image = cv2.imread(file_path)
        #     #[1000,1000,3]
        #     image = image[image.shape[0]//4:-image.shape[0]//4, image.shape[1]//4:-image.shape[1]//4, :]
        #     if self.image_width != image.shape[0]:
        #         image = cv2.resize(image, (self.image_height, self.image_width))
        #     #image = cv2.resize(image[100:-100,100:-100,:], (self.image_width, self.image_width),
        #     #                   interpolation=cv2.INTER_LINEAR)
        #     #[128,128,3]

        #     frames_np.append(np.array(image, dtype=np.float32) / 255.0)
        #     frames_file_name.append(filename)
        #     #if len(frames_np) % 100 == 0: print len(frames_np)
        #     #if len(frames_np) % 1000 == 0: break
        # # is it a begin index of sequence
        # indices = []
        # index = 0
        # print ('gen index')
        # while index + intervel * self.seq_len - 1 < len(frames_file_name):
        #     # 'S11_Discussion_1.54138969_000471.jpg'
        #     # ['S11_Discussion_1', '54138969_000471', 'jpg']
        #     start_infos = frames_file_name[index].split('.')
        #     end_infos = frames_file_name[index+intervel*(self.seq_len-1)].split('.')
        #     if start_infos[0] != end_infos[0]:
        #         index += 1
        #         continue
        #     start_video_id, start_frame_id = start_infos[1].split('_')
        #     end_video_id, end_frame_id = end_infos[1].split('_')
        #     if start_video_id != end_video_id:
        #         index += 1
        #         continue
        #     if int(end_frame_id) - int(start_frame_id) == 5 * (self.seq_len - 1) * intervel:
        #         indices.append(index)
        #     index += 10
        dataset = np.load(self.data_dir+'/human_train.npz')
        self.data = dataset['a']
        indices = dataset['b']
        print("there are " + str(len(self.data)) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        #np.savez_compressed('human_train', a = frames_np,b = indices)
        return self.data, indices

    def __getitem__(self, index):
        #print(index)
        if self.with_mask:
            all_masks = create_random_shape_with_random_motion(
                self.lb, imageHeight=self.image_height, imageWidth=self.image_width)
            all_masks = np.stack([np.expand_dims(x, 2) for x in all_masks], axis=0)/ 255.0  
        idx_id = self.indeces[index]
        #self.data = np.array(self.data) #��listת��Ϊarray
        inputs = self.data[idx_id:idx_id+self.lb]  #[t,h,w,c]
        targets = self.data[idx_id+self.lb+1:idx_id+self.lb+1+self.pred_h]
        if self.with_mask:
            return torch.tensor(inputs.transpose(0,3,1,2),dtype=torch.float), torch.tensor(targets.transpose(0,3,1,2),dtype=torch.float), torch.tensor(all_masks.transpose(0,3,1,2),dtype=torch.float)
        else:
            return torch.tensor(inputs.transpose(0,3,1,2),dtype=torch.float), torch.tensor(targets.transpose(0,3,1,2),dtype=torch.float)
        

    def __len__(self):
        return len(self.indeces)



class Human_Occulusion_Test(Dataset):
    def __init__(self, data_dir, nt_cond, nt_pred, mask):
        super(Human_Occulusion_Test,self).__init__()
        self.data_dir = data_dir
        self.pred_h = nt_pred
        self.lb = nt_cond
        self.image_height = 128
        self.image_width = 128
        self.seq_len = nt_cond + nt_pred
        self.data, self.indeces = self.load_data(data_dir)
        self.mean = 0
        self.std = 1
        self.with_mask = mask
        self.mask =  np.load(self.data_dir+'/human_mask.npz')['arr_0']
        #self.mask =  np.load('/data/dataset/Human/human_mask.npz')['arr_0']
        # self._to_tensors = transforms.Compose([
        #     Stack(),
        #     #ToTorchFormatTensor(),
        # ])
    def load_data(self, paths):
        # data_dir = paths
        # intervel = 2
        
        # frames_np = []
        # scenarios = ['Walking']
        # subjects = ['S9', 'S11']
        # #subjects = ['S9']
        # _path = data_dir
        # print('Using dataset Human3.6M(Test) now')
        # #print ('load data...', _path)
        # filenames = os.listdir(_path)
        # filenames.sort()
        # print ('data size ', len(filenames))
        # frames_file_name = []
        # for filename in filenames:
        #     fix = filename.split('.')
        #     fix = fix[0]
        #     subject = fix.split('_')
        #     scenario = subject[1]
        #     subject = subject[0]
        #     if subject not in subjects or scenario not in scenarios:
        #         continue
        #     file_path = os.path.join(_path, filename)
        #     #image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        #     image = cv2.imread(file_path)
        #     #[1000,1000,3]
        #     image = image[image.shape[0]//4:-image.shape[0]//4, image.shape[1]//4:-image.shape[1]//4, :]
        #     if self.image_width != image.shape[0]:
        #         image = cv2.resize(image, (self.image_height, self.image_width))
        #     #image = cv2.resize(image[100:-100,100:-100,:], (self.image_width, self.image_width),
        #     #                   interpolation=cv2.INTER_LINEAR)
        #     #[128,128,3]

        #     frames_np.append(np.array(image, dtype=np.float32) / 255.0)
        #     frames_file_name.append(filename)
        #     #if len(frames_np) % 100 == 0: print len(frames_np)
        #     #if len(frames_np) % 1000 == 0: break
        # # is it a begin index of sequence
        # indices = []
        # index = 0
        # print ('gen index')
        # while index + intervel * self.seq_len - 1 < len(frames_file_name):
        #     # 'S11_Discussion_1.54138969_000471.jpg'
        #     # ['S11_Discussion_1', '54138969_000471', 'jpg']
        #     start_infos = frames_file_name[index].split('.')
        #     end_infos = frames_file_name[index+intervel*(self.seq_len-1)].split('.')
        #     if start_infos[0] != end_infos[0]:
        #         index += 1
        #         continue
        #     start_video_id, start_frame_id = start_infos[1].split('_')
        #     end_video_id, end_frame_id = end_infos[1].split('_')
        #     if start_video_id != end_video_id:
        #         index += 1
        #         continue
        #     if int(end_frame_id) - int(start_frame_id) == 5 * (self.seq_len - 1) * intervel:
        #         indices.append(index)
        #     index += 5
        
        # print("there are " + str(len(indices)) + " sequences")
        # # data = np.asarray(frames_np)
        # self.data = frames_np
        # print("there are " + str(len(self.data)) + " pictures")
        # #np.savez_compressed('human_test', a = frames_np,b = indices)
        dataset = np.load(self.data_dir+'/human_test.npz')
        self.data = dataset['a']
        indices = dataset['b']
        print("there are " + str(len(self.data)) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return self.data, indices

    def __getitem__(self, index):
        #file_name = '/data/datasets/Human3.6M/test_masks/'
        idx_id = self.indeces[index]
        #self.data = np.array(self.data) #��listת��Ϊarray
        #inputs = self.data[idx_id:idx_id+self.lb].transpose(0,3,1,2)
        inputs = self.data[idx_id:idx_id+self.lb]  #[t,h,w,c]
        #cv2.imwrite(file_name +'train_org'+'.png', (inputs[0,:,:,:]*255))
        targets = self.data[idx_id+self.lb+1:idx_id+self.lb+1+self.pred_h]
        if self.with_mask:
            masks = self.mask[:,index,...]
        # for i in range(4):
        #     cv2.imwrite(file_name +'mask'+ str(i) + '.png', (masked_frames[i]*255))
        if self.with_mask:
            return torch.tensor(inputs.transpose(0,3,1,2),dtype=torch.float), torch.tensor(targets.transpose(0,3,1,2),dtype=torch.float), torch.tensor(masks.transpose(0,3,1,2),dtype=torch.float)
        else:
            return torch.tensor(inputs.transpose(0,3,1,2),dtype=torch.float), torch.tensor(targets.transpose(0,3,1,2),dtype=torch.float)
        

    def __len__(self):
        return len(self.indeces)



def load_data(
        batch_size, val_batch_size, mask,
        data_root, num_workers):
    
    train_set = Human_Occulusion_Train(data_dir=data_root, nt_cond=4, nt_pred=4, mask=mask)
    test_set = Human_Occulusion_Test(data_dir=data_root, nt_cond=4, nt_pred=4, mask=mask)
    
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std