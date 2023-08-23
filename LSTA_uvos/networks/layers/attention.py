import torch
from torch import nn
import torch.nn.functional as F
import math

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import SyncBatchNorm as norm_layer

def shape_change(x):
    """
    input size: [batch, channel, height, width]
    output size: [batch, height*width, channel]
    """
    b,c,_,_ = x.size()
    out = x.view(b,c,-1).permute(0,2,1)
    return out

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b,c,h,w = x.size()
        device = x.device
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, m in zip(x, mask):
          m[: img.shape[1], :img.shape[2]] = False
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # pos.requires_grad=False
        return pos


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class PerformerSimple(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.head_cnt = head_cnt
        self.epsilon = 1e-8  # for stable in division

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        # print(x.shape)
        # print(self.w.shape)
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def forward(self, q,k,v):
        """
        q: tensor, size=[batch, channel, height, width]
        k: tensor, size=[batch, channel, height, width]
        v: tensor, size=[batch, channel, height, width]
        """
        b,c,h,w =q.size()
        residual = q
        q = shape_change(q)
        k = shape_change(k)
        v = shape_change(v)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        y = y.permute(0,2,1).view(b,c,h,w)
        # out = torch.cat([y, residual], dim=1)
        h,w = y.shape[-2:]
        y = torch.layer_norm(y, (h,w))

        return y

class EAHead(nn.Module):
    def __init__(self, c):
        super(EAHead, self).__init__()
        self.k = 32 
        self.first_conv = nn.Conv2d(c, c, 1)
        self.k_linear = nn.Conv1d(c, self.k, 1, bias=False)
        self.v_linear = nn.Conv1d(self.k, c, 1, bias=False)

    def forward(self, x):
        idn = x[:]
        b, c, h, w = x.size()
        x = self.first_conv(x)
        x = x.view(b, c, -1) # b, c, n 
        attn = self.k_linear(x) # b, c, n
        attn = torch.softmax(attn, dim=-1)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-9) 
        x = self.v_linear(attn) # b, c, n 
        x = x.view(b, c, h, w)
        x = x + idn 
        return x  

class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super(External_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x
# class EANet(Module):
#     def __init__(self, num_classes=21, output_stride=16):
#         super(EANet(), self).__init__()
#         self.backbone = resnet50(output_stride)
#         self.mid_conv = nn.Conv(2048, 512, 1)
#         self.head = EAHead(512)
#         self.final_conv = nn.Conv(512, num_classes, 1)

#     def execute(self, x):
#         imsize = x.shape 
#         x = self.backbone(x)  
#         x = self.mid_conv(x)
#         x = self.head(x) 
#         x = self.final_conv(x)
#         x = nn.resize(x, size=(imsize[2], imsize[3]), mode='bilinear')
#         return x 

class TemporalPatchAttentionV9(nn.Module):
    # def __init__(self,in_dim,out_dim, kernel=16,stride=8) -> None:
    def __init__(self,in_dim,out_dim, kernel=8,stride=4) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj_prev = nn.Conv2d(in_dim, out_dim, 1)
        # self.proj_curr = nn.Conv2d(in_dim, out_dim, 1)
        # self.attn_fn = PerformerSimple(out_dim, out_dim)

    def split(self,x):
        """
        x [batch, channel, height, width]
        =>[batch*N,channel,kernel, kernel]
        
        """
        batch,channel,h,w = x.size()
        
        patch_split = nn.Unfold(self.kernel,stride=self.stride)
        x = patch_split(x)
        
        x = x.view(batch,channel,self.kernel*self.kernel,-1)
        x = x.permute(0,3,1,2) # [b,n,c,k*k]torch.Size([2, 1508, 128, 49])

        batch,N,channel,_ = x.size()
        # N = x.size(1)
        x = x.reshape(batch*N,channel,-1).permute(0,2,1)

        return x
    
    def merge(self,y, origin_size):
        """
        y  [batch,N,C,K,K]
        """
        y = y.permute(0,2,3,4,1) #=>[batch,channel,k,k,n]
        batch,channel,k,k,n = y.size()
        y = y.reshape(batch,channel*k*k,n)
        z = nn.Fold(origin_size,kernel_size=self.kernel, stride=self.stride)(y)
        return z

    def forward(self, q, k,v):
        original_shape = v.shape
        b_origin,c_origin,h_orign, w_origin = original_shape
        # b,c,h,w = feat_curr.size()
        # q = self.proj_curr(feat_curr)
        # k = v = self.proj_prev(feat_prev)
        
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        
        batch,_,channel = q.shape
        attn_map = torch.matmul(q, k.permute(0,2,1))
        # [b,hw,c]x [b,c,hw]->[b,hw,hw]
        attn_map = F.softmax((channel ** -.5) * attn_map, dim=-1)
        z = torch.matmul(attn_map, v).view(batch, self.kernel, self.kernel, channel)
        # print(z.shape)
        # z = self.attn_fn(q,k,v)
        _,kernel,_,channel= z.size()
        z = z.reshape(b_origin,-1,kernel,kernel,channel).permute(0,1,4,2,3)
        z = self.merge(z, (h_orign,w_origin))
        
        return z 

class TemporalPatchAttention(nn.Module):
    def __init__(self,in_dim,out_dim, kernel,stride):
        super(TemporalPatchAttention, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj_prev = nn.Conv2d(in_dim, out_dim, 1)
        self.proj_curr = nn.Conv2d(in_dim, out_dim, 1)        
        self.attn_fn = PerformerSimple(out_dim, out_dim)
        
    def split(self,x):
        """
        x [batch, channel, height, width]
        =>[batch*N,channel,kernel, kernel]
        
        """
        batch,channel,h,w = x.size()
        
        patch_split = nn.Unfold(self.kernel,stride=self.stride)
        x = patch_split(x)
        
        x = x.view(batch,channel,self.kernel*self.kernel,-1)
        x = x.permute(0,3,1,2) # [b,n,c,k*k]torch.Size([2, 1508, 128, 49])

        batch,N,channel,_ = x.size()
        N = x.size(1)
        x = x.reshape(batch*N,channel,self.kernel,self.kernel)
        return x
    
    def merge(self,y, origin_size):
        """
        y  [batch,N,C,K,K]
        """
        y = y.permute(0,2,3,4,1) #=>[batch,channel,k,k,n]
        batch,channel,k,k,n = y.size()
        y = y.view(batch,channel*k*k,n)
        z = nn.Fold(origin_size,kernel_size=self.kernel, stride=self.stride)(y)
        return z
    
    def forward(self, feat_curr, feat_prev):
        b,c,h,w = feat_curr.size()
        q = self.proj_curr(feat_curr)
        k = v = self.proj_prev(feat_prev)
        
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        
        z = self.attn_fn(q,k,v)
        _,c,k,_= z.size()
        z = z.view(b,-1,c,k,k)
        z = self.merge(z, (h,w))
        
        return z
        

class Performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.head_cnt = head_cnt
        self.epsilon = 1e-8  # for stable in division

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def forward(self, q,k,v):
        """
        q: tensor, size=[batch, channel, height, width]
        k: tensor, size=[batch, channel, height, width]
        v: tensor, size=[batch, channel, height, width]
        """
        b,c,h,w =q.size()
        residual = q
        q = shape_change(q)
        k = shape_change(k)
        v = shape_change(v)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        y = y.permute(0,2,1).view(b,c,h,w)
        out = torch.cat([y, residual], dim=1)
        h,w = out.shape[-2:]
        out = torch.layer_norm(out, (h,w))

        return out


def spatial_attentions_s(q,k,v):
    """
    q: tensor, size=[batch, channel, height, width]
    k: tensor, size=[batch, channel, height, width]
    v: tensor, size=[batch, channel, height, width]

    return:
    out: tensor, size=[batch, channel, height, width]
    """
    out = None

    batch, channel, h, w = q.shape
    batch, channel, h, w = q.shape
    M = h * w
    # q = shape_change(q)
    # k = shape_change(k)
    # v = shape_change(v)

    attn_map = torch.matmul(q, k.permute(0,2,1))
    # [b,hw,c]x [b,c,hw]->[b,hw,hw]
    attn_map = F.softmax((channel ** -.5) * attn_map, dim=-1)
    out = torch.matmul(attn_map, v).permute(0, 2, 1).view(batch, channel, h, w)
    # [b,hw,hw]x[b,hw,c]->[b,hw,c]->[b,c,hw]->[b,c,h,w]
    return out

def spatial_attention(q,k,v):
    """
    q: tensor, size=[batch, channel, height, width]
    k: tensor, size=[batch, channel, height, width]
    v: tensor, size=[batch, channel, height, width]

    return:
    out: tensor, size=[batch, channel, height, width]
    """
    out = None

    batch, channel, h, w = q.shape
    M = h * w
    q = shape_change(q)
    k = shape_change(k)
    v = shape_change(v)

    attn_map = torch.matmul(q, k.permute(0,2,1))
    # [b,hw,c]x [b,c,hw]->[b,hw,hw]
    attn_map = F.softmax((channel ** -.5) * attn_map, dim=-1)
    out = torch.matmul(attn_map, v).permute(0, 2, 1).view(batch, channel, h, w)
    # [b,hw,hw]x[b,hw,c]->[b,hw,c]->[b,c,hw]->[b,c,h,w]
    return out


def channel_attention(q,k,v):
    """
    q: tensor, size=[batch, channel, height, width]
    k: tensor, size=[batch, channel, height, width]
    v: tensor, size=[batch, channel, height, width]

    return:
    out: tensor, size=[batch, channel, height, width]
    """
    # feat_attn = model.attn_fn(feat_curr, feat_prev, feat_prev)

    out = None

    batch, channel, h, w = v.shape
    M = h * w
    q = shape_change(q)     #[b,hw,c]
    k = shape_change(k)     #[b,hw,c]
    v = shape_change(v)     #[b,hw,c]
    attn_map = torch.matmul(q.permute(0,2,1), k)        #[b,c,hw]x[b,hw,c]->[b,c,c]
    attn_map = F.softmax((M ** -.5) * attn_map, dim=-1)
    out = torch.matmul(attn_map, v.permute(0,2,1)).view(batch, channel, h, w)
    # [b,c,c]x[b,c,hw]->[b,c,hw]->[b,c,hw]->[b,c,h,w]
    return out