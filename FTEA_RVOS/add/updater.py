from einops.einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LanguageGuidedMemoryUpdater(nn.Module):
    def __init__(self, vdim_in, l_dim_in, dim_out, kernel_size):
        super().__init__()
        # 1. input language feature and memory feature
        # 2. attention 
        # 3. retrive language related visual feature
        # 4. retrive visual related language feature
        # language feature should change according to frame index
        self.padding = int((kernel_size - 1) / 2)

        self.input_size_v  = vdim_in
        self.hidden_size = dim_out
        self.reset_gate_v = nn.Conv2d(vdim_in + dim_out, dim_out, kernel_size, padding=self.padding)
        self.update_gate_v = nn.Conv2d(vdim_in + dim_out, dim_out, kernel_size, padding=self.padding)
        self.out_gate_v = nn.Conv2d(vdim_in + dim_out, dim_out, kernel_size, padding=self.padding)

        self.input_size_l  = l_dim_in
        self.reset_gate_l = nn.Conv1d(l_dim_in + dim_out, dim_out, kernel_size=1)
        self.update_gate_l = nn.Conv1d(l_dim_in + dim_out, dim_out, kernel_size=1)
        self.out_gate_l = nn.Conv1d(l_dim_in + dim_out, dim_out, kernel_size=1)



        nn.init.xavier_normal_(self.reset_gate_v.weight)
        nn.init.xavier_normal_(self.update_gate_v.weight)
        nn.init.xavier_normal_(self.out_gate_v.weight)
        nn.init.constant_(self.reset_gate_v.bias, 0.)
        nn.init.constant_(self.update_gate_v.bias, 0.)
        nn.init.constant_(self.out_gate_v.bias, 0.)

        nn.init.xavier_normal_(self.reset_gate_l.weight)
        nn.init.xavier_normal_(self.update_gate_l.weight)
        nn.init.xavier_normal_(self.out_gate_l.weight)
        nn.init.constant_(self.reset_gate_l.bias, 0.)
        nn.init.constant_(self.update_gate_l.bias, 0.)
        nn.init.constant_(self.out_gate_l.bias, 0.)  

    def forward(self, feat_vis, feat_lang, prev_state_v=None, prev_state_l=None):
        """[summary]

        Args:
            feat_vis ([tensor]): [size (b, c, h, w)]
            feat_lang ([tensor]): [size (b, n, c)]
            prev_state_v ([type], optional): [description]. Defaults to None.
            prev_state_l ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if prev_state_v is None:
            b,_,h,w = feat_vis.shape
            prev_state_v = feat_vis
        if prev_state_l is None:
            b, n,_ = feat_lang.shape
            prev_state_l = feat_lang

        # data size is [batch, channel, height, width]
        stacked_inputs_v = torch.cat([feat_vis, prev_state_v], dim=1)
        stacked_inputs_l = torch.cat([feat_lang, prev_state_l], dim=2)
        update_v = self.update_gate_v(stacked_inputs_v)
        reset_v = self.reset_gate_v(stacked_inputs_v)
        stacked_inputs_l = stacked_inputs_l.transpose(2, 1)
        update_l = self.update_gate_l(stacked_inputs_l)
        update_l = update_l.transpose(2, 1)
        reset_l = self.reset_gate_l(stacked_inputs_l)
        reset_l = reset_l.transpose(2, 1)
        b, c, h, w = update_v.shape
        query = update_v.view(b, c, h*w).transpose(2, 1)
        b, n, c = update_l.shape
        key = update_l.transpose(2, 1)
        attention_map_v = torch.bmm(query, key) / math.sqrt(c) # b, hw, n
        # print(attention_map_v.shape)
        related_v = torch.max_pool1d(attention_map_v, n).transpose(2, 1).view(b, 1, h, w) # [b, 1, h, w]
        related_l = torch.max_pool1d(attention_map_v.transpose(2, 1), h*w) # [b, n, 1]

        # # learn more ?
        # related_v = self.weight_gate_v(related_v)
        # related_l = self.weight_gate_l(related_l.transpose(2,1)).transpose(2,1)

        # print(related_v.shape, related_l.shape)
        update_v_ = torch.sigmoid(update_v)
        reset_v_ = torch.sigmoid(reset_v)
        out_inputs_v = torch.tanh(self.out_gate_v(torch.cat([feat_vis * torch.sigmoid(related_v), prev_state_v * reset_v_], dim=1)))
        new_state_v = prev_state_v * (1 - update_v_) + out_inputs_v * update_v_

        update_l_ = torch.sigmoid(update_l)
        reset_l_ = torch.sigmoid(reset_l)
        x_in = torch.cat([feat_lang * torch.sigmoid(related_l), prev_state_l * reset_l_], dim=2)
        x_in = x_in.transpose(2,1)
        out_inputs_l = torch.tanh(self.out_gate_l(x_in))
        out_inputs_l = out_inputs_l.transpose(2, 1)
        new_state_l = prev_state_l * (1 - update_l_) + out_inputs_l * update_l_
        return new_state_v, new_state_l


class DynamicAdaptation(nn.Module):
    """[single branch, only for language]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, vdim_in, l_dim_in, dim_out, kernel_size=3, dropout=0.0):
        super().__init__()
        # 1. input language feature and memory feature
        # 2. attention 
        # 3. retrive language related visual feature
        # 4. retrive visual related language feature
        # language feature should change according to frame index
        self.padding = int((kernel_size - 1) / 2)

        self.input_size_v  = vdim_in
        self.hidden_size = dim_out
        # self.reset_gate_v = nn.Conv2d(vdim_in + dim_out, dim_out, kernel_size, padding=self.padding)
        self.update_gate_v = nn.Conv2d(vdim_in + dim_out, dim_out, kernel_size, padding=self.padding)
        # self.out_gate_v = nn.Conv2d(vdim_in + dim_out, dim_out, kernel_size, padding=self.padding)

        self.input_size_l  = l_dim_in
        self.reset_gate_l = nn.Linear(l_dim_in + dim_out, dim_out)
        self.update_gate_l = nn.Linear(l_dim_in + dim_out, dim_out)
        self.out_gate_l = nn.Linear(l_dim_in + dim_out, dim_out)

        # self.weight_gate_v = nn.Conv2d(1, 1, kernel_size=1)
        # self.weight_gate_l = nn.Conv1d(1, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.update_gate_v.weight)
        nn.init.constant_(self.update_gate_v.bias, 0.)
        # nn.init.constant_(self.out_gate_v.bias, 0.)

        nn.init.orthogonal_(self.reset_gate_l.weight)
        nn.init.orthogonal_(self.update_gate_l.weight)
        nn.init.orthogonal_(self.out_gate_l.weight)
        nn.init.constant_(self.reset_gate_l.bias, 0.)
        nn.init.constant_(self.update_gate_l.bias, 0.)
        nn.init.constant_(self.out_gate_l.bias, 0.)  

    def forward(self, feat_vis, feat_lang, prev_state_v=None, prev_state_l=None):
        """[summary]

        Args:
            feat_vis ([tensor]): [size (b, c, h, w)]
            feat_lang ([tensor]): [size (b, n, c)]
            prev_state_v ([type], optional): [description]. Defaults to None.
            prev_state_l ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if prev_state_v is None:
            b,_,h,w = feat_vis.shape
            prev_state_v = feat_vis
        if prev_state_l is None:
            b, n,_ = feat_lang.shape
            prev_state_l = feat_lang

        # data size is [batch, channel, height, width]
        stacked_inputs_v = torch.cat([feat_vis, prev_state_v], dim=1)
        stacked_inputs_l = torch.cat([feat_lang, prev_state_l], dim=2)
        update_v = self.update_gate_v(stacked_inputs_v)

        # stacked_inputs_l = stacked_inputs_l.transpose(2, 1)
        update_l = self.update_gate_l(stacked_inputs_l)
        # update_l = update_l.transpose(2, 1)
        reset_l = self.reset_gate_l(stacked_inputs_l)
        # reset_l = reset_l.transpose(2, 1)
        b, c, h, w = update_v.shape
        query = update_v.view(b, c, h*w).transpose(2, 1)
        b, n, c = update_l.shape
        key = update_l.transpose(2, 1)
        attention_map_v = torch.bmm(query, key) / math.sqrt(c) # b, hw, n

        related_l = torch.max_pool1d(attention_map_v.transpose(2, 1), h*w) # [b, n, 1]

        update_l_ = torch.sigmoid(update_l)
        reset_l_ = torch.sigmoid(reset_l)
        x_in = torch.cat([feat_lang * torch.sigmoid(related_l), prev_state_l * reset_l_], dim=2)
        # x_in = x_in.transpose(2,1)
        out_inputs_l = torch.tanh(self.out_gate_l(x_in))
        # out_inputs_l = out_inputs_l.transpose(2, 1)
        new_state_l = prev_state_l * (1 - update_l_) + out_inputs_l * update_l_

        return new_state_l


class DynamicAttentiveAdaptation(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.conv_q = nn.Conv2d(c_in, c_out, 1)
        self.conv_k = nn.Conv2d(c_in, c_out, 1)
        self.conv_v = nn.Conv2d(c_in, c_out, 1)

        self.conv_w = nn.Conv2d(c_in, c_out, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.linear_q = nn.Linear(c_in, c_out)
        self.linear_k = nn.Linear(c_in, c_out)
        self.linear_v = nn.Linear(c_in, c_out)
        self.linear_w = nn.Linear(c_in, c_out)
        self.ln = nn.LayerNorm(c_out)
    
    def forward(self, f_v, f_l):
        """[summary]

        Args:
            f_v ([tensor]): [shape = (t b c h w)]
            f_l ([tensor]): [shape = (t b s c)]
        """
        t, b, _, h,w = f_v.shape
        f_v_outs = []
        f_l_outs = []
        for tt in range(t):
            mask = [i for i in range(t)]
            mask.pop(tt)
            f_cur = f_v[tt,:]
            f_ref = f_v.index_select(0, torch.tensor(mask).to(f_v.device))
            q = self.conv_q(f_cur)
            k = self.conv_k(rearrange(f_ref, 't b c h w -> (t b) c h w', t=t-1))
            v = self.conv_v(rearrange(f_ref, 't b c h w -> (t b) c h w', t=t-1))

            w1 = self.linear_w(f_l[tt,:])
            w1 = torch.sigmoid(torch.max(w1, dim=1)[0])
            w1 = repeat(w1, 'b c-> b c h w', h=h, w=w)

            q = q * w1
            q = rearrange(q, 'b c h w->b (h w) c')
            k = rearrange(k, '(t b) c h w-> b c (t h w)', t=t-1, b=b)
            v = rearrange(v, '(t b) c h w-> b (t h w) c', t=t-1, b=b)
            attention_map = torch.bmm(q, k) / math.sqrt(q.shape[-1]) # b * hw * thw
            attention_map = torch.softmax(attention_map, dim=-1)
            out = torch.bmm(attention_map, v)
            f_v_out = rearrange(out, 'b (h w) c-> b c h w', h=h, w=w)
            f_v_out = F.relu(self.bn(f_v_out))
            f_v_outs.append(f_v_out.unsqueeze(0))

            lq = self.linear_q(f_l[tt,:])
            lk = self.linear_k(f_l[tt,:])
            lv = self.linear_v(f_l[tt,:])

            w2 = self.conv_w(f_cur)
            w2 = torch.sigmoid(F.adaptive_max_pool2d(w2,(1,1))).squeeze(-1).squeeze(-1).unsqueeze(1)
            # print(lq.shape, w2.shape)
            # orch.Size([1, 12, 256]) torch.Size([256, 1]) 
            lq = lq * w2
            attention_map = torch.bmm(lq, lk.transpose(2, 1)) / math.sqrt(lq.shape[-1]) # b * s * s
            attention_map = torch.softmax(attention_map, dim=-1)
            out = torch.bmm(attention_map, lv)
            out = F.relu(self.ln(out))
            f_l_outs.append(out.unsqueeze(0))
        f_l_outs = torch.cat(f_l_outs, dim=0)
        f_v_outs = torch.cat(f_v_outs, dim=0)

        return f_v_outs, f_l_outs
                        


if __name__ == "__main__":
    T = 8
    x = torch.rand(T,1, 512, 16, 28)
    y = torch.rand(T,1, 20, 512)

    # m = LanguageGuidedMemoryUpdater(vdim_in=512, l_dim_in=512, dim_out=512, kernel_size=3)
    # x_p, y_p = None, None
    # for i in range(T):
    #     x_p, y_p = m(x[:,i,:], y[:,i,:], x_p, y_p)
    #     # x_p, y_p = x, y
    # print(torch.mean(x_p), torch.var(x_p))

    # m = LanguageGuidedMemoryUpdaterV4(vdim_in=512, l_dim_in=512, dim_out=512, kernel_size=3)
    # x_p, y_p = None, None
    # for i in range(T):
    #     y_p = m(x[:,i,:], y[:,i,:], x_p, y_p)
    #     # x_p, y_p = x, y
    #     x_p = x[:,i,:]
    # print(torch.mean(x_p), torch.var(x_p))
    m = DynamicAttentiveAdaptation(512, 256)
    a, b = m(x, y)
    print(a.shape, b.shape)
