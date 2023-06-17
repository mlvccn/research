import os
import time
import argparse
import torch
from torch.autograd.grad_mode import F
from torchsummary import summary
from thop import profile

from model import *
from inf_model import *

def compute_speed(model, input_size, gpu=0, iteration=100):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # model.cuda(gpu)
    # model.eval()

    # with torch.no_grad():
    input = torch.randn(*input_size).cuda(gpu)
    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--size", type=str, default="480,640", help="input size of model")
    parser.add_argument("--size", type=str, default="1280,720", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--n_class', type=int, default=5)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='DPAdd50_EGCFM_k5_addHead')
    parser.add_argument('--layers', type=int, default=18)
    args = parser.parse_args()

    h, w = map(int, args.size.split(','))
    if args.model_name == 'MFNet' or args.model_name == 'SegNet':
        model = eval(args.model_name)(n_class=args.n_class)
    else:
        model = eval(args.model_name)(n_class=args.n_class, layers=args.layers)
    gpu = 2
    model.cuda(gpu)
    model.eval()
    # compute the fps
    compute_speed(model, (args.batch_size, args.num_channels, h, w), gpu, iteration=args.iter)
    
    rgb = torch.randn(1, 3, 1280, 720)
    thermal = torch.randn(1, 1, 1280, 720)
    input = torch.cat((rgb, thermal), dim=1).cuda(gpu)

    # summary(model, input_size=(4, 480, 640), device='cuda')
    total_ops, total_params  = profile(model, inputs=(input, ), verbose=False)
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("%s | %.2f | %.2f" % (args.model_name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))
