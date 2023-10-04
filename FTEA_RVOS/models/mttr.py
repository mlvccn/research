"""
Modified from DETR https://github.com/facebookresearch/detr
"""

from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn
from misc import NestedTensor
from models.backbone import init_backbone
from models.matcher import build_matcher
from models.segmentation import FPNSpatialDecoder
from models.multimodal_transformer import MultimodalTransformer
from models.criterion import SetCriterion
from models.postprocessing import A2DSentencesPostProcess, ReferYoutubeVOSPostProcess
from einops import rearrange
import math
from add.spatial_decoder import FPNSpatialDecoderV3, FPNSpatialDecoderV4, FPNSpatialDecoderV5
from add.instance_reason import QueryReasoning

class MTTR(nn.Module):
    """ The main module of the Multimodal Tracking Transformer """
    def __init__(self, num_queries, mask_kernels_dim=8, aux_loss=False, **kwargs):
        """
        Parameters:
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MTTR can detect in a single image. In our paper we use 50 in all settings.
            mask_kernels_dim: dim of the segmentation kernels and of the feature maps outputted by the spatial decoder.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = init_backbone(**kwargs)
        self.transformer = MultimodalTransformer(**kwargs)
        d_model = self.transformer.d_model
        self.is_referred_head = nn.Linear(d_model, 2)  # binary 'is referred?' prediction head for object queries
        self.instance_kernels_head = MLP(d_model, d_model, output_dim=mask_kernels_dim, num_layers=2)
        self.obj_queries = nn.Embedding(num_queries, d_model)  # pos embeddings for the object queries
        self.vid_embed_proj = nn.Conv2d(self.backbone.layer_output_channels[-1], d_model, kernel_size=1)
        self.spatial_decoderv4 = FPNSpatialDecoderV5(d_model, self.backbone.layer_output_channels[:-1][::-1], mask_kernels_dim,object_candidates=num_queries)
        self.aux_loss = aux_loss

        self.w = torch.randn(mask_kernels_dim, mask_kernels_dim)
        self.loss_mat = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(mask_kernels_dim), requires_grad=True)
        self.mask_kernels_dim = mask_kernels_dim

    def fix_param(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(False)

    def forward(self, samples: NestedTensor, valid_indices, text_queries):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: Batched frames of shape [time x batch_size x 3 x H x W]
               - samples.mask: A binary mask of shape [time x batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_is_referred": The reference prediction logits for all queries.
                                     Shape: [time x batch_size x num_queries x 2]
               - "pred_masks": The mask logits for all queries.
                               Shape: [time x batch_size x num_queries x H_mask x W_mask]
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        backbone_out = self.backbone(samples)
        # keep only the valid frames (frames which are annotated):
        # (for example, in a2d-sentences only the center frame in each window is annotated).
        for layer_out in backbone_out:
            layer_out.tensors = layer_out.tensors.index_select(0, valid_indices)
            layer_out.mask = layer_out.mask.index_select(0, valid_indices)
        bbone_final_layer_output = backbone_out[-1]
        vid_embeds, vid_pad_mask = bbone_final_layer_output.decompose()

        T, B, _, _, _ = vid_embeds.shape
        vid_embeds = rearrange(vid_embeds, 't b c h w -> (t b) c h w')
        vid_embeds = self.vid_embed_proj(vid_embeds)
        vid_embeds = rearrange(vid_embeds, '(t b) c h w -> t b c h w', t=T, b=B)

        transformer_out = self.transformer(vid_embeds, vid_pad_mask, text_queries, self.obj_queries.weight)
        # hs is: [L, T, B, N, D] where L is number of decoder layers
        # 3, 1, 2, 50, 256]
        # vid_memory is: [T, B, D, H, W]
        # txt_memory is a list of length T*B of [S, C] where S might be different for each sentence
        # encoder_middle_layer_outputs is a list of [T, B, H, W, D]
        hs, vid_memory, txt_memory = transformer_out
        # orch.Size([3, 1, 2, 50, 256]) torch.Size([1, 2, 256, 20, 36])

        vid_memory = rearrange(vid_memory, 't b d h w -> (t b) d h w')
        bbone_middle_layer_outputs = [rearrange(o.tensors, 't b d h w -> (t b) d h w') for o in backbone_out[:-1][::-1]]
        instance_kernels = self.instance_kernels_head(hs)  # [L, T, B, N, C]
        decoded_frame_features = self.spatial_decoderv4(vid_memory, bbone_middle_layer_outputs, instance_kernels)



        decoded_frame_features = rearrange(decoded_frame_features, '(t b) d h w -> t b d h w', t=T, b=B)
        output_masks = torch.einsum('ltbnc,tbchw->ltbnhw', instance_kernels, decoded_frame_features)
        outputs_is_referred = self.is_referred_head(hs)  # [L, T, B, N, 2]

        layer_outputs = []
        for pm, pir in zip(output_masks, outputs_is_referred):
            layer_out = {'pred_masks': pm,
                         'pred_is_referred': pir}
            layer_outputs.append(layer_out)
        out = layer_outputs[-1]  # the output for the last decoder layer is used by default
        if self.aux_loss:
            out['aux_outputs'] = layer_outputs[:-1]
        out['kernel_loss'] = self.object_kernel_loss_w(instance_kernels[-1,...])
        return out

    def object_kernel_loss(self, hs, alpha=0.007):
        """[summary]

        Args:
            hs ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.001, value of loss is about 0.5

        Returns:
            [type]: [description]
        """
        t,b,n,c = hs.shape
        hs = rearrange(hs, 't b n c-> (t b ) n c')
        score_map = torch.bmm(hs, hs.transpose(2, 1)) / math.sqrt(c)
        ltb = t * b
        identiy_matrix = torch.eye(n).unsqueeze(0).repeat(ltb, 1, 1).to(hs.device)
        loss = torch.norm(score_map-identiy_matrix, p='fro', dim=(1,2), keepdim=True)
        loss = loss.view(t, b, 1, 1)
        return loss * alpha

    def object_kernel_loss_w(self, hs, alpha=0.007):
        """[summary]

        Args:
            hs ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.001, value of loss is about 0.5

        Returns:
            [type]: [description]
        """
        t,b,n,c = hs.shape
        hs = rearrange(hs, 't b n c-> (t b ) n c')

        score_map = torch.einsum('bnc,cd->bnd',hs, self.loss_mat) # (t b) n c, c c --> (tb)
        score_map = torch.bmm(score_map, hs.transpose(2, 1)) / math.sqrt(c)
        ltb = t * b
        identiy_matrix = torch.eye(n).unsqueeze(0).repeat(ltb, 1, 1).to(hs.device)
        loss = torch.norm(score_map-identiy_matrix, p='fro', dim=(1,2), keepdim=True)
        loss = loss.view(t, b, 1, 1)
        return loss * alpha

    def object_kernel_loss_w_all(self, hs_all, alpha=0.007):
        """[summary]

        Args:
            hs ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.001, value of loss is about 0.5

        Returns:
            [type]: [description]
        """
        loss_all = 0
        for l in range(hs_all.shape[0]):
            hs = hs_all[l,...]
            t,b,n,c = hs.shape
            hs = rearrange(hs, 't b n c-> (t b ) n c')
            score_map = torch.einsum('bnc,cd->bnd',hs, self.loss_mat) # (t b) n c, c c --> (tb)
            score_map = torch.bmm(score_map, hs.transpose(2, 1)) / math.sqrt(c)
            ltb = t * b
            identiy_matrix = torch.eye(n).unsqueeze(0).repeat(ltb, 1, 1).to(hs.device)
            loss = torch.norm(score_map-identiy_matrix, p='fro', dim=(1,2), keepdim=True)
            loss = loss.view(t, b, 1, 1)
            loss_all = loss + loss_all
        loss_all = loss_all / hs_all.shape[0]
        return loss_all * alpha

    def object_kernel_loss_w_norm(self, hs, alpha=0.007):
        """[summary]

        Args:
            hs ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.001, value of loss is about 0.5

        Returns:
            [type]: [description]
        """
        t,b,n,c = hs.shape
        hs = rearrange(hs, 't b n c-> (t b ) n c')
        hsnorm = torch.norm(hs, p=2, dim=-1, keepdim=True) + 1e-6
        hs = hs.div(hsnorm.expand_as(hs))
        score_map = torch.bmm(hs, hs.transpose(2, 1)) # tb * n * n
        ltb = t * b
        identiy_matrix = torch.eye(n).unsqueeze(0).repeat(ltb, 1, 1).to(hs.device)
        loss = torch.norm(score_map-identiy_matrix, p='fro', dim=(1,2), keepdim=True)
        loss = loss.view(t, b, 1, 1)
        return loss * alpha

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = args.device
    model = MTTR(**vars(args))
    matcher = build_matcher(args)
    weight_dict = {'loss_is_referred': args.is_referred_loss_coef,
                   'loss_dice': args.dice_loss_coef,
                   'loss_sigmoid_focal': args.sigmoid_focal_loss_coef}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef)
    criterion.to(device)
    if args.dataset_name == 'a2d_sentences' or args.dataset_name == 'jhmdb_sentences':
        postprocessor = A2DSentencesPostProcess()
    elif args.dataset_name == 'ref_youtube_vos':
        postprocessor = ReferYoutubeVOSPostProcess()
    else:
        assert False, f'postprocessing for dataset: {args.dataset_name} is not supported'
    return model, criterion, postprocessor
