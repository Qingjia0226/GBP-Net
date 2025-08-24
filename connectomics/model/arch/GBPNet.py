from __future__ import print_function, division
import math
import matplotlib.pyplot as plt
from ..block import *
from ..utils import model_init
from mamba_ssm import Mamba
import torch

args = {'num_attention_heads': 4,
        'L': 4,
        'D': 96,
        'mlp_dim': 512,
        'dropout_rate': 0.}

block_dict = {
    'residual': BasicBlock3d,
    'residual_pa': BasicBlock3dPA,
    'residual_se': BasicBlock3dSE,
    'residual_se_pa': BasicBlock3dPASE,
}

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.num_attention_heads = args['num_attention_heads']
        self.attention_head_size = int(args['D'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args['D'], self.all_head_size)
        self.key = nn.Linear(args['D'], self.all_head_size)
        self.value = nn.Linear(args['D'], self.all_head_size)

        self.out = nn.Linear(args['D'], args['D'])
        self.attn_dropout = nn.Dropout(args['dropout_rate'])
        self.proj_dropout = nn.Dropout(args['dropout_rate'])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x2)
        mixed_value_layer = self.value(x2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(args['D'], args['mlp_dim'])
        self.fc2 = nn.Linear(args['mlp_dim'], args['D'])
        self.act_fn = torch.nn.functional.gelu  # torch.nn.functional.relu
        self.dropout = nn.Dropout(args['dropout_rate'])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.ffn_norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.ffn = Mlp()
        self.norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.attn = Attention()

    def forward(self, x1, x2):
        identity = x1
        x1 = self.attention_norm(x1)
        x2 = self.norm(x2)
        x1 = self.attn(x1, x2)
        x1 = x1 + identity

        identity = x1
        x1 = self.ffn_norm(x1)
        x1 = self.ffn(x1)
        x1 = x1 + identity

        return x1


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(args['D'], eps=1e-6)
        for _ in range(args['L']):
            layer = Block()
            self.layer.append(layer)

    def forward(self, x1, x2):
        for layer_block in self.layer:
            x1 = layer_block(x1, x2)
        encoded = self.encoder_norm(x1)
        return encoded



class Embeddings(nn.Module):
    def __init__(self, in_channels, hidden_size, n_voxels):
        super(Embeddings, self).__init__()
        k_size = 1
        self.patch_embeddings = nn.Conv3d(in_channels, out_channels=hidden_size, kernel_size=k_size, stride=k_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_voxels, hidden_size))
        self.dropout = nn.Dropout(args['dropout_rate'])

    def forward(self, x):
        # print(x.shape[0])
        # print(self.position_embeddings.shape)

        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Transformer(nn.Module):
    def __init__(self, in_channels1, in_channels2, n_voxels, hidden_size=args['D']):
        super(Transformer, self).__init__()
        self.embed1 = Embeddings(in_channels1, hidden_size, n_voxels)
        self.embed2 = Embeddings(in_channels2, hidden_size, n_voxels)
        self.encoder = Encoder()

    def forward(self, x_l, x_g):
        w=x_l.shape[-1]
        embed1 = self.embed1(x_l)
        embed2 = self.embed2(x_g)  # (B, n_patch, hidden//2)
        encoded = self.encoder(embed1, embed2)
        B, n_patch, hidden = encoded.size()
        encoded = encoded.permute(0, 2, 1)
        encoded = encoded.contiguous().view(B, hidden, 2, w, w)
        return encoded




class MambaLayer(nn.Module):
    def __init__(self, dim=32, d_state=16, d_conv=4, expand=2):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        out = self.forward_patch_token(x)

        return out

class FusionLayer(nn.Module):
    def __init__(self, dim=32, mlp_dim=256):
        super().__init__()

        self.mam1= MambaLayer(dim=32)
        self.c1= nn.Conv3d(32,16, kernel_size=3,padding=(1, 1, 1))
        self.bn1=nn.BatchNorm3d(16)
        self.mam2 = MambaLayer(dim=16)
        self.c2= nn.Conv3d(16,16, kernel_size=3,padding=(1, 1, 1))
        self.bn2=nn.BatchNorm3d(16)
        

    def forward(self, l, s):
        x = torch.cat((l, s), dim=1)
        x = self.mam1(x) + x
        x = self.c1(x)
        x=self.bn1(x)

        x = self.mam2(x) + x
        x = self.c2(x)
        x= self.bn2(x)

        return x 


class GBPNet(nn.Module):
    def __init__(self, block_type='residual', in_channel: int = 1, out_channel: int = 3,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 pad_mode='replicate', act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 pooling: bool = False, return_feats: Optional[list] = None, **kwargs):
        super().__init__()
        self.depth = len(filters)
        self.return_feats = return_feats
        isotropy = [False, True, True, True, True]
        is_isotropic = False
        block = self.block_dict[block_type]
        self.pooling = pooling
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}
        kernel_size_io, padding_io = self._get_kernal_size(is_isotropic, io_layer=True)
  
        self.conv_in_s = conv3d_norm_act(in_channel, filters[0], (1, 5, 5), padding=(0, 2, 2), **self.shared_kwargs)
        self.conv_out_s = conv3d_norm_act(filters[0]+ 1, out_channel, (1, 5, 5), bias=True, padding=(0, 2, 2),
                                          pad_mode=pad_mode, act_mode='none', norm_mode='none')

        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size_io,
                                       padding=padding_io, **self.shared_kwargs)
        self.conv_out = conv3d_norm_act(filters[0] + 1, out_channel, kernel_size_io, bias=True,
                                        padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')


        # encoding path CUN
        self.down_layers = nn.ModuleList()

        for i in range(self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[i])
            previous = max(0, i - 1)
            stride = self._get_stride(isotropy[i], previous, i)
            layer = nn.Sequential(
                self._make_pooling_layer(isotropy[i], previous, i),
                conv3d_norm_act(filters[previous], filters[i], kernel_size,
                                stride=stride, padding=padding, **self.shared_kwargs),
                block(filters[i], filters[i],  **self.shared_kwargs))
            self.down_layers.append(layer)

        # encoding path HRUN
        self.down_layers_h = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[i])
            previous = max(0, i - 1)
            stride = self._get_stride(isotropy[i], previous, i)

            layer = nn.Sequential(
                self._make_pooling_layer(isotropy[i], previous, i),
                conv3d_norm_act(filters[previous], filters[i], kernel_size,
                                stride=stride,  padding=padding, **self.shared_kwargs),
                block(filters[i], filters[i],  **self.shared_kwargs))
            self.down_layers_h.append(layer)

        self.transformer = Transformer(96, 96, n_voxels=2 * 4 * 4, hidden_size=96)

        # decoding path
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[j])
            layer = nn.ModuleList([
                conv3d_norm_act(filters[j], filters[j - 1], kernel_size,
                                padding=padding, **self.shared_kwargs),
                block(filters[j - 1], filters[j - 1], **self.shared_kwargs)])
            self.up_layers.append(layer)

        # decoding path
        self.up_layers_s = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[j])
            layer = nn.ModuleList([
                conv3d_norm_act(filters[j], filters[j - 1], kernel_size,
                                padding=padding, **self.shared_kwargs),
                block(filters[j - 1], filters[j - 1], **self.shared_kwargs)])
            self.up_layers_s.append(layer)
        # initialization

        self.fuse = FusionLayer()
        model_init(self, mode='kaiming')

    
    def forward(self, x_h, x_c):

        x_re = torch.clone(x_c)
        x_res = torch.clone(x_h)
        x_c = self.conv_in(x_c)
        down_x_c = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            x_c = self.down_layers[i](x_c)
            down_x_c[i] = x_c
        x_c = self.down_layers[-1](x_c)

        x_h = self.conv_in_s(x_h)
        down_x_h = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            x_h = self.down_layers_h[i](x_h)
            down_x_h[i] = x_h
        x_h = self.down_layers_h[-1](x_h)
        x_h = self.transformer(x_h, x_c)

        for j in range(self.depth - 1):
            i = self.depth - 2 - j
            x_c = self.up_layers[i][0](x_c)
            x_c = self._upsample_add(x_c, down_x_c[i])
            x_c = self.up_layers[i][1](x_c)

            x_h = self.up_layers_s[i][0](x_h)
            x_h = self._upsample_add(x_h, down_x_h[i])
            x_h = self.up_layers_s[i][1](x_h)

            if j == 3:
                B, C, D, W, H = x_c.shape
                ROI = x_c[:, :, D // 3: 2 * D // 3, W // 3: 2 * W // 3, H // 3: 2 * H // 3]
                ROI = F.interpolate(ROI, size=[D, W, H], mode='trilinear', align_corners=False)
                x_h = self.fuse(ROI, x_h)

        x_h = self.conv_out_s(torch.cat((x_res, x_h), dim=1))
        x_c = self.conv_out(torch.cat((x_re, x_c), dim=1))

        return x_h, x_c

    def _upsample_add(self, x, y):
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y.shape[2:], mode='trilinear',
                          align_corners=align_corners)
        return x + y

    def _get_kernal_size(self, is_isotropic, io_layer=False):
        if io_layer:  # kernel and padding size of I/O layers
            if is_isotropic:
                return (5, 5, 5), (2, 2, 2)
            return (1, 5, 5), (0, 2, 2)

        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)

    def _get_stride(self, is_isotropic, previous, i):
        if self.pooling or previous == i:
            return 1

        return self._get_downsample(is_isotropic)

    def _get_downsample(self, is_isotropic):
        if not is_isotropic:
            return (1, 2, 2)
        return 2

    def _make_pooling_layer(self, is_isotropic, previous, i):
        if self.pooling and previous != i:
            kernel_size = stride = self._get_downsample(is_isotropic)
            return nn.MaxPool3d(kernel_size, stride)

        return nn.Identity()

