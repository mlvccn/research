import argparse
from turtle import st
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parametersp
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='mnist_3fft', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=1, type=int)
    
    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj','human','kth','kitti'])
    parser.add_argument('--num_workers', default=8, type=int)
    
    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj  
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=512, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=3, type=int)
    parser.add_argument('--groups', default=8, type=int)
    
    # Training parameters
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--mask', default=1, type=int, help='Input frames with mask')
    parser.add_argument('--recover_loss', default=1, type=int, help='Using recover loss')
    parser.add_argument('--lamda', default=0.5, type=float, help='hyperparameter of recover loss')
    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)