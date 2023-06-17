
from ast import arg
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import SimVP
from tqdm import tqdm
from API import *
from utils import *
import scipy.stats as st
import cv2
import torch.nn.functional as F

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        
        self._preparation()
        print_log(output_namespace(self.args))

        #self._get_data()
        self._select_optimizer()
        self._select_criterion()
        # if args.pretrained_model:
        #     self._load()
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)   # '0,1'
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device
    
    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T,groups=args.groups, mask=args.mask).to(self.device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        print_log(print_model_parm_nums(self.model))

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)
    
    def _load(self, name=''):
        self.model.load_state_dict(torch.load('./results/Debug/checkpoint.pth'))
        f = open(os.path.join(self.checkpoints_path, 'checkpoint' + '.pkl'),'rb')
        state = pickle.load(f)
        self.scheduler.load_state_dict(state)
        
    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            if config['mask']:
                print('Training with mask')
                for batch_x, batch_y, mask in train_pbar:
                    self.optimizer.zero_grad()
                    batch_x, batch_y,mask = batch_x.to(self.device), batch_y.to(self.device),mask.to(self.device)
                    batch_x = (batch_x * (1 - mask).float())
                    pred_y,re_x = self.model(batch_x)
                    loss_1 = self.criterion(pred_y, batch_y)
                    if config['recover_loss']:
                        loss_2 = self.criterion(re_x, batch_x)
                    else:
                        loss_2 = 0
                    loss = loss_1 + config['lamda'] * loss_2
                    train_loss.append(loss.item())
                    train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
            else:
                print('Training without mask')
                for batch_x, batch_y in train_pbar:
                    self.optimizer.zero_grad()
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_y,re_x = self.model(batch_x)
                    loss = self.criterion(pred_y,batch_y)
                    train_loss.append(loss.item())
                    train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
                    
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
            
            train_loss = np.average(train_loss)


            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader,config)
                    if epoch % (args.log_step * 10) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Learning rate: {1:.4f} | Train Loss: {2:.4f} Vali Loss: {3:.4f}\n".format(
                    epoch + 1, self.optimizer.param_groups[0]['lr'], train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)
        best_model_path = self.path + '/' + 'checkpoint.pth'          
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model                
        
    def vali(self, vali_loader, config):
        
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        if config['mask']:
            for i, (batch_x, batch_y, mask) in enumerate(vali_pbar):
                if i * batch_x.shape[0] > 1000:
                    break
                batch_x, batch_y, mask  = batch_x.to(self.device), batch_y.to(self.device), mask.to(self.device)
                batch_x_mask = (batch_x * (1 - mask).float())
                pred_y,_ = self.model(batch_x_mask)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                    pred_y, batch_y], [preds_lst, trues_lst]))

                loss = self.criterion(pred_y, batch_y)
                
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())
        else:
            for i, (batch_x, batch_y) in enumerate(vali_pbar):
                if i * batch_x.shape[0] > 1000:
                    break
                batch_x, batch_y  = batch_x.to(self.device), batch_y.to(self.device)
                pred_y,_ = self.model(batch_x)

                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                    pred_y, batch_y], [preds_lst, trues_lst]))

                loss = self.criterion(pred_y, batch_y)

            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss
    
    def test(self, args):
        config = args.__dict__
        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        if config['mask']:
            for batch_x, batch_y, mask in self.test_loader:
                batch_x, mask = batch_x.to(self.device), mask.to(self.device)
                batch_x = (batch_x * (1 - mask).float())
                pred_y,_ = self.model(batch_x)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
        else:
            test_pbar = tqdm(self.test)
            for batch_x, batch_y in test_pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                pred_y,_ = self.model(batch_x)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])
        
        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse
