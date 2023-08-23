import torch
from torch import nn


#############################################################
######    Paper:https://arxiv.org/abs/1710.05941
#######   Code: 
@torch.jit.script
def swish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.sigmoid(input)


class Swish(nn.Module):
    '''
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return swish(input)

############################################################################################################
class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    # https://github.com/nmaac/acon
    """
    def __init__(self, width):
        super().__init__()

        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        return (self.p1 * x - self.p2 * x) * self.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

