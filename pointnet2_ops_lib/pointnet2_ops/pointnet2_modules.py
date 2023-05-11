from typing import List, Optional, Tuple

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from torch_scatter import scatter_max, scatter_mean, scatter_sum
import numpy as np
import random




def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        # print(f'features.shape = {features.shape}')
        
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            # print(f'new_features.shape= {new_features.shape}')
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


# begin{lcx}
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, activation=True, dim=1, groups=1):
        super().__init__()
        if activation:
            activate = nn.LeakyReLU
        if bn == True:
            if dim == 1:
                conv = nn.Conv1d
                bn = nn.BatchNorm1d
            elif dim == 2:
                conv = nn.Conv2d
                bn = nn.BatchNorm2d
            else:
                raise ValueError
            if not isinstance(out_channels, (list, tuple)):
                out_channels = [out_channels]
            layers = []

            for oc in out_channels:
                if not activation:
                    layers.extend([
                        conv(in_channels, oc, 1, groups=groups, bias=False),
                        bn(oc),
                    ])
                else:
                    layers.extend([
                        conv(in_channels, oc, 1, groups=groups, bias=False),
                        bn(oc),
                        activate(0.1, True)
                    ])
                in_channels = oc
        else:
            if dim == 1:
                conv = nn.Conv1d
            elif dim == 2:
                conv = nn.Conv2d
            else:
                raise ValueError
            if not isinstance(out_channels, (list, tuple)):
                out_channels = [out_channels]
            layers = []
            if not activation:
                for oc in out_channels:
                    layers.extend([
                        conv(in_channels, oc, 1, groups=groups, bias=False),
                    ])
                    in_channels = oc
            else:
                for oc in out_channels:
                    layers.extend([
                        conv(in_channels, oc, 1, groups=groups, bias=False),
                        activate(0.1, True)
                    ])
                    in_channels = oc

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return 

class APPblock(nn.Module):
    r"""
    Parameters
    ----------
    resolution: denote as r, the coord of point cloud P is in (0, r)
        integer number
    in_channels: c_in
        integer number
    out_channels: c_out
        integer number
    """

    def __init__(self, in_channels, out_channels, down_sample_rate=4, auxiliary_rate=0, baseop='concat', use_feat=True, use_pos=True):
        super().__init__()

        self.nsample = 1  # 1-nn
        # two rate: 1. rate for auxiliary point; 2. for output feature aggregation sample rate
        self.down_sample_rate = down_sample_rate
        assert self.down_sample_rate > 0
        if auxiliary_rate != 0:
            self.auxiliary_rate = auxiliary_rate
        else:
            self.auxiliary_rate = self.down_sample_rate//2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.baseop = baseop.split(',')
        self.use_feat = use_feat
        self.use_pos = use_pos
        # self.secondstage = secondstage
        print(
            f'Initializing APP block with down sample rate [{down_sample_rate}], auxiliary rate [{auxiliary_rate}], in->out: {in_channels}->{out_channels}')

        self.pos_enc = SharedMLP(3, in_channels, bn=True, activation=True)
        self.skip = SharedMLP(in_channels + in_channels, out_channels, groups=1)
        # self.channel_mixing = SharedMLP(in_channels, in_channels, groups=2)

        if 'concat' in self.baseop or 'expconcat' in self.baseop:
            self.cat_att = SharedMLP(in_channels+in_channels, in_channels, bn=False, activation=False)
        elif 'exp' in self.baseop or 'expdiff' in self.baseop or 'exp2' in self.baseop:
            self.pos_att = SharedMLP(in_channels, in_channels, bn=False, activation=False)
        elif 'sin' in self.baseop or 'cos' in self.baseop:
            self.cat_att = SharedMLP(in_channels+in_channels, in_channels, bn=False, activation=False)
        elif 'sindiff' in self.baseop or 'cosdiff' in self.baseop:
            self.cat_att = SharedMLP(in_channels+in_channels, in_channels, bn=False, activation=False)
        elif 'mulsin' in self.baseop or 'mulcos' in self.baseop:
            pass
        elif 'sinweight' in self.baseop or 'cosweight' in self.baseop or 'sinwcosweight' in self.baseop:
            assert self.use_feat or self.use_pos
            if self.use_feat and self.use_pos:
                self.pos_att = SharedMLP(in_channels+in_channels, in_channels, bn=False, activation=False)
            else:
                self.pos_att = SharedMLP(in_channels, in_channels, bn=False, activation=False)
        elif 'sinweight_x' in self.baseop or 'cosweight_x' in self.baseop:
            self.pos_att = SharedMLP(in_channels+in_channels, in_channels, bn=False, activation=False)
        elif 'noagg' in self.baseop:
            self.cat_att = SharedMLP(in_channels+in_channels, in_channels, bn=False, activation=False)
        else:
            assert self.baseop in ['exp', 'concat', 'expconcat', 'exp2', 'expdiff', 'sin', 'cos', 'sindiff', 'cosdiff',  'mulsin', 'mulcos', 'sinweight', 'cosweight', 'sinweight_x', 'cosweight_x', 'noagg']
        
        
        
        self.saved = False
        self.vis_cnt = 0


    def forward(self, coords, features, iterative=1, pt=None, st=None, pos_enc=None):
        r"""
        Parameters
        ----------
        coords: [B, 3, N]
        features: [B, C, N]
        """

        #### indexing
        B, _, N = coords.shape
        section = N // self.auxiliary_rate
        if section == 0:
            iterative = 1
        
        start_loc = random.randint(0, N)
        
        if self.use_pos:
            pos_enc = self.pos_enc(coords[:,:3,:])
        else:
            pos_enc = None
        features_set = []
        
        for i in range(min(self.auxiliary_rate, iterative)):
            # random sample
            if section == 0:
                group_idx = (torch.zeros(1)).cuda().view([1, -1, 1]).repeat([B, N, 1]).contiguous().long()
            else:
                idx = (torch.arange(start_loc+i*section, start_loc + (i+1)*section) % N).cuda()
                auxiliary_point = coords[:, :3, idx]
                
                # pf = torch.cat([coords, features*3.0/self.in_channels], dim=1)
                # auxiliary_pf = pf[:, :, idx]
                # group_idx = pointnet2_utils.one_nn_pf(pf.permute(0, 2, 1).contiguous(),
                #                     auxiliary_pf.permute(0, 2, 1).contiguous()) # .unsqueeze(dim=-1)
                # group_idx = pointnet2_utils.one_nn_pf(pf.permute(0, 2, 1).contiguous(),
                #                                         auxiliary_pf.permute(0, 2, 1).contiguous())
                # print(f'group_idx: {group_idx.shape}')
                # scat_group_idx = group_idx.permute(0, 2, 1).clone().detach().long()

                group_idx = pointnet2_utils.chamfer_distance(coords.permute(0, 2, 1).contiguous(),
                                    auxiliary_point.permute(0, 2, 1).contiguous()).unsqueeze(dim=-1)

            '''
            # fps
            # fps_idx = pointnet2_utils.furthest_point_sample(coords.contiguous(), section)
            # auxiliary_point = pointnet2_utils.gather_operation(coords.contiguous(), fps_idx)
            '''
            
            
            # randomshuffle
            # if section != 0:
            #     group_idx = (torch.randint(low=0, high=section, size=(section,))).cuda().view([1, -1, 1]).repeat([B, self.auxiliary_rate, 1]).contiguous().long()
            # else:
            #     group_idx = (torch.zeros(1)).cuda().view([1, -1, 1]).repeat([B, N, 1]).contiguous().long()
            # print(f'group_idx: {group_idx.shape}, {group_idx.max()}')
            
            # auxiliary_pe = torch.rand_like(auxiliary_pe)*1000
            # auxiliary_pe = auxiliary_pe * torch.rand(1).cuda()
            # aux_pe = pointnet2_utils.gather_operation(
            #     auxiliary_pe.contiguous(),
            #     group_idx.squeeze(-1).int())

            # pos_enc  = pos_enc - aux_pe

            scat_group_idx = group_idx.permute(0, 2, 1).clone().detach().long()  # (B, 1, N)
            for op in self.baseop:
                features_weight = self._aggregator(pos_enc, features, group_idx, scat_group_idx, op)
                features_set.append(features_weight)
        
        features_agg = torch.stack(features_set, dim=1).max(dim=1).values
        
        fused_features = self.skip(torch.cat([features, features_agg], dim=1))


        ##### block down sample
        # random sample
        if self.down_sample_rate == 1:
            return coords, fused_features
            
        centroids = coords[:, :, :N // (self.down_sample_rate)].contiguous()
        half_downsampled_one_nn_idx = pointnet2_utils.chamfer_distance(coords.permute(0, 2, 1).contiguous(),
                                                                centroids[:,:3].permute(0,2,1).contiguous()).unsqueeze(dim=-1)
        
        new_scat_group_idx = half_downsampled_one_nn_idx.permute(0, 2, 1).clone().detach().long()
        down_sampled_point_features, _ = scatter_max(fused_features, new_scat_group_idx.repeat(1, self.out_channels, 1),
                                                     dim=2)
        
        return centroids, down_sampled_point_features

    def _aggregator(self, pos_enc, features, group_idx, scat_group_idx, baseop='cosweight'):
        if baseop == 'concat':
            features_weight = self.cat_att(torch.cat([features, pos_enc], dim=1))

            #### push && pull START ####
            aggregate_features = scatter_mean(features_weight,
                                                     scat_group_idx.repeat(1, features_weight.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push && pull END ####

            features_weight = corresponding_centroid_features - features_weight
        
        elif baseop == 'expconcat':    
            features_cat = self.cat_att(torch.cat([features, pos_enc], dim=1))
            features_weight = torch.exp(features_cat)
            
            #### push && pull START ####
            aggregate_features = scatter_mean(features_weight,
                                                     scat_group_idx.repeat(1, features_weight.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push && pull END ####

            features_weight = corresponding_centroid_features / features_weight

        elif baseop == 'exp':
            pos_att = self.pos_att(pos_enc / torch.tensor(self.in_channels*1.0).cuda().sqrt())
            weight = torch.exp(pos_att)
            # print(f'features.shape = {features.shape}, weight.shape->{weight.shape}')
            features_weight = features * weight
            
            #### # push & pull START ####
            aggregate_features = scatter_mean(features_weight,
                                                     scat_group_idx.repeat(1, features_weight.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            features_weight = corresponding_centroid_features / weight
        
        #### e^(x+y) + e^(-x-y) = 0.5 * ( (e^x+e^(-x)) * (e^y+e^(-y)) + (e^x-e^(-x)) * (e^y-e^(-y)) )
        elif baseop == 'expdiff':
            pos_att = self.pos_att(pos_enc / torch.tensor(self.in_channels*1.0).cuda().sqrt())
            weight = torch.exp(pos_att)
            weight_r = 1.0 / weight
            features_add = features * (weight + weight_r)
            features_sub = features * (weight - weight_r)

            features_weight = torch.cat([features_add, features_sub], dim=1)
            
            #### push & pull START ####
            features_weight = torch.cat([features_weight], dim=1)
            aggregate_features = scatter_mean(features_weight,
                                                     scat_group_idx.repeat(1, features_weight.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####
            
            features_add = corresponding_centroid_features[:,:self.in_channels, :]
            features_sub = corresponding_centroid_features[:,self.in_channels:, :]

            features_weight = 0.5*(features_add*(weight + weight_r) + features_sub*(weight_r-weight))
        
        #### 2^(x+y) = 2^x+2^y
        elif baseop == 'exp2':
            pos_att = self.pos_att(pos_enc)
            weight = 2**pos_att
            features_weight = features * weight
            
            #### push & pull START ####
            features_weight = torch.cat([features_weight], dim=1)
            aggregate_features = scatter_mean(features_weight,
                                                     scat_group_idx.repeat(1, features_weight.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            features_weight = corresponding_centroid_features / weight

        #### sin(x+y) = sin(x)cos(y)+cos(x)sin(y) 
        elif baseop == 'sin':
            features_weight = self.cat_att(torch.cat([features, pos_enc], dim=1))
            zeros_weight = self.cat_att(torch.cat([torch.ones_like(features).cuda(), pos_enc], dim=1))
            sin_feat = torch.sin(features_weight)
            cos_feat = torch.cos(features_weight)
            sin_feat_pull = torch.sin(zeros_weight)
            cos_feat_pull = torch.cos(zeros_weight)
            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = corresponding_centroid_sin_features*cos_feat_pull + corresponding_centroid_cos_features*(-sin_feat_pull)
       
        #### sin(x+y) = sin(x)cos(y)+cos(x)sin(y) 
        elif baseop == 'sindiff':
            features_weight = self.cat_att(torch.cat([features, pos_enc], dim=1))
            sin_feat = torch.sin(features_weight)
            cos_feat = torch.cos(features_weight)
            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####            
            aggregate_features = scatter_mean(sin_cos_cat, scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1), dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####
            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = corresponding_centroid_sin_features*cos_feat + corresponding_centroid_cos_features*(-sin_feat)
 
        #### sin(x+y) = sin(x)cos(y)+cos(x)sin(y) 
        elif baseop == 'mulsin':
            sin_weight = torch.sin(pos_enc)
            cos_weight = torch.cos(pos_enc)
            sin_feat = sin_weight * features
            cos_feat = cos_weight * features

            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = corresponding_centroid_sin_features*cos_weight + corresponding_centroid_cos_features*(-sin_weight)
        
        #### sin(x+y) = sin(x)cos(y)+cos(x)sin(y) 
        elif baseop == 'sinweight':
            # pos_att = self.pos_att(pos_enc / torch.tensor(self.in_channels*1.0).cuda().sqrt())
            # aux_pe = torch.rand_like(pos_enc)*1000
            # pos_enc = pos_enc - aux_pe
            # pos_enc = pos_enc #  / torch.tensor(self.in_channels*1.0).cuda().sqrt()
            
            # if self.use_feat:
            #     pos_att = self.pos_att(torch.cat([pos_enc, features], dim=1))
            # else:
            pos_att = self.pos_att(pos_enc)
            
            sin_weight = torch.sin(pos_att)
            cos_weight = torch.cos(pos_att)
            sin_feat = sin_weight * features
            cos_feat = cos_weight * features

            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = corresponding_centroid_sin_features*cos_weight + corresponding_centroid_cos_features*(-sin_weight)
            
        #### sin(x+y)+x+y = sin(x)cos(y)+cos(x)sin(y)+x+y
        elif baseop == 'sinweight_x':
            # pos_att = self.pos_att(pos_enc / torch.tensor(self.in_channels*1.0).cuda().sqrt())
            pos_enc = pos_enc / torch.tensor(self.in_channels*1.0).cuda().sqrt()
            pos_att = self.pos_att(torch.cat([pos_enc, features], dim=1))
            # pos_att = self.pos_att(features)
            
            sin_weight = torch.sin(pos_att)
            cos_weight = torch.cos(pos_att)
            sin_feat = sin_weight * features
            cos_feat = cos_weight * features

            sin_cos_cat = torch.cat([sin_feat, cos_feat, features], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:2*self.in_channels,:]
            corresponding_centroid_features = corresponding_centroid_features[:,2*self.in_channels:,:]

            features_weight = corresponding_centroid_sin_features*cos_weight + corresponding_centroid_cos_features*(-sin_weight) + corresponding_centroid_features - features
        
        #### cos(x+y) = cos(x)cos(y)-sin(x)sin(y) 
        elif baseop == 'cos':
            features_weight = self.cat_att(torch.cat([features, pos_enc], dim=1))
            zeros_weight = self.cat_att(torch.cat([torch.zeros_like(features).cuda(), pos_enc], dim=1))
            sin_feat = torch.sin(features_weight)
            cos_feat = torch.cos(features_weight)
            sin_feat_pull = torch.sin(zeros_weight)
            cos_feat_pull = torch.cos(zeros_weight)
            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = corresponding_centroid_cos_features*cos_feat_pull - corresponding_centroid_sin_features*(-sin_feat_pull)
       

        #### cos(x+y) = cos(x)cos(y)-sin(x)sin(y) 
        elif baseop == 'cosdiff':
            features_weight = self.cat_att(torch.cat([features, pos_enc], dim=1))
            sin_feat = torch.sin(features_weight)
            cos_feat = torch.cos(features_weight)
            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = corresponding_centroid_cos_features*cos_feat - corresponding_centroid_sin_features*(-sin_feat)
 
        #### cos(x+y) = cos(x)cos(y)-sin(x)sin(y) 
        elif baseop == 'mulcos':
            sin_weight = torch.sin(pos_enc)
            cos_weight = torch.cos(pos_enc)
            sin_feat = sin_weight * features
            cos_feat = cos_weight * features

            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = -corresponding_centroid_sin_features*sin_weight + corresponding_centroid_cos_features*(-cos_weight)
        
        #### cos(x+y) = cos(x)cos(y)-sin(x)sin(y) 
        elif baseop == 'cosweight':
            # div = torch.tensor(self.in_channels*1.0).cuda().sqrt()
            # aux_pe = aux_pe - torch.rand_like(aux_pe)*1000
            # pos_enc = pos_enc - aux_pe
            if self.use_pos and (not self.use_feat):
                pos_att = self.pos_att(pos_enc)
            elif self.use_feat and (not self.use_pos):
                pos_att = self.pos_att(torch.cat([features], dim=1))
            else:
                pos_enc = pos_enc / self.in_channels **0.5 # torch.tensor(self.in_channels*1.0).cuda().sqrt()
                pos_att = self.pos_att(torch.cat([pos_enc, features], dim=1))
            
            sin_weight = torch.sin(pos_att)
            cos_weight = torch.cos(pos_att)
            sin_feat = sin_weight * features
            cos_feat = cos_weight * features

            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = corresponding_centroid_cos_features*cos_weight + corresponding_centroid_sin_features*sin_weight
            
        #### cos(x+y)+x+y = cos(x)cos(y)-sin(x)sin(y) + x + y
        elif baseop == 'cosweight_x':
            pos_enc = pos_enc / torch.tensor(self.in_channels*1.0).cuda().sqrt()
            pos_att = self.pos_att(torch.cat([pos_enc, features], dim=1))
            
            sin_weight = torch.sin(pos_att)
            cos_weight = torch.cos(pos_att)
            sin_feat = sin_weight * features
            cos_feat = cos_weight * features

            sin_cos_cat = torch.cat([sin_feat, cos_feat, features], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            corresponding_centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            corresponding_centroid_cos_features = corresponding_centroid_features[:,self.in_channels:2*self.in_channels,:]
            corresponding_centroid_features = corresponding_centroid_features[:,2*self.in_channels:,:]
            features_weight = -corresponding_centroid_sin_features*sin_weight + corresponding_centroid_cos_features*(-cos_weight) + corresponding_centroid_features - features

        #### sin(x+y) = sin(x)cos(y)+cos(x)sin(y) 
        #### cos(x+y) = cos(x)cos(y)-sin(x)sin(y) 
        elif baseop == 'sinwcosweight':
            # pos_att = self.pos_att(pos_enc / torch.tensor(self.in_channels*1.0).cuda().sqrt())
            pos_enc = pos_enc / torch.tensor(self.in_channels*1.0).cuda().sqrt()
            pos_att = self.pos_att(torch.cat([pos_enc, features], dim=1))
            # pos_att = self.pos_att(features)
            
            sin_weight = torch.sin(pos_att)
            cos_weight = torch.cos(pos_att)
            sin_feat = sin_weight * features
            cos_feat = cos_weight * features

            sin_cos_cat = torch.cat([sin_feat, cos_feat], dim=1)
            #### push & pull START ####
            aggregate_features = scatter_mean(sin_cos_cat,
                                                     scat_group_idx.repeat(1, sin_cos_cat.shape[1], 1),
                                                     dim=2)  # (B, C, M)
            corresponding_centroid_features = pointnet2_utils.gather_operation(
                aggregate_features.contiguous(),
                group_idx.squeeze(
                    -1).int())  # [B, C, N]
            #### push & pull END ####

            centroid_sin_features = corresponding_centroid_features[:,:self.in_channels,:]
            centroid_cos_features = corresponding_centroid_features[:,self.in_channels:,:]

            features_weight = centroid_sin_features*cos_weight + centroid_cos_features*(-sin_weight) + \
                centroid_cos_features*cos_weight + centroid_sin_features*sin_weight
        #### 
        elif baseop == 'noagg':
            features_weight = self.cat_att(torch.cat([features, pos_enc], dim=1))
            
        
        return features_weight


class APPFPModule(nn.Module):
    r"""Propagates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(APPFPModule, self).__init__()
        # self.mlp = SharedMLP(mlp)
        self.mlp = build_shared_mlp(mlp)
        print(
            f'Initializing APPFPModule, in->out: {mlp[0]}->{mlp[-1]}')
        # self.mlp2 = build_shared_mlp(mlp2, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats, one_nn_idx=None, pre_knn=False):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        # 1nn
        if known is not None:
            idx = pointnet2_utils.chamfer_distance(unknown, known)
            interpolated_feats = pointnet2_utils.one_interpolate(
                known_feats, idx
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)
    
class UpSample(nn.Module):
    r"""Upsample the feature
    """

    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, unknown, known, known_feats):
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        
        Returns
        -------
        interpolated_features : torch.Tensor
        
        """
        
        idx = pointnet2_utils.chamfer_distance(unknown, known)
        interpolated_feats = pointnet2_utils.one_interpolate(known_feats, idx)
            
        return interpolated_feats


if __name__ == "__main__":
    print("Start debug")
    import time
    for i in range(7, 18):
        B = 4
        N = 2**i # 131072
        C = 32
        # region component debug
        p = torch.rand(B, N, 3).cuda()
        x = torch.rand(B, N//16, 3).cuda()
        torch.cuda.synchronize()
        start = time.time()
        group_idx = pointnet2_utils.chamfer_distance(x, p) 
        torch.cuda.synchronize()
        end = time.time()
        print(f'2^{i}: 1nn: {(end-start)*1000}')
        # torch.cuda.synchronize()
        # start = time.time()
        # group_idx = pointnet2_utils.furthest_point_sample(p, N//4) 
        # torch.cuda.synchronize()
        # end = time.time()
        # print(f'fps: {end-start} for point: {p.size()}')
        # endregion component debug

        # region APP debug
        # model = APPblock(C, C, down_sample_rate=4).cuda()
        # for _ in range(3):
        #     p = torch.randn(B, 3, N).cuda()
        #     feat = torch.randn(B, C, N).cuda()
        #     xyz, f = model(p, feat)
        # endregion APP debug

        # region profier
        # # warm-up
        # for _ in range(5):
        #     start = time.time()
        #     p = torch.randn(B, 3, N).cuda()
        #     feat = torch.randn(B, C, N).cuda()
        #     xyz, f = model(p, feat)
        #     torch.cuda.synchronize()
        #     end = time.time()
        #     print('Time:{}ms'.format((end-start)*1000))

        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        #     xyz, f = model(p, feat)
        # print(prof.table())
        # endregion profier

        # region pointnet debug
        # model = nn.ModuleList().cuda()
        # model.append(
        #         PointnetSAModule(
        #             npoint=512,
        #             radius=0.2,
        #             nsample=64,
        #             mlp=[3, 64, 128],
        #             # mlp=[3, 64, 64, 128],
        #             use_xyz=True,
        #             use_APP=True,
        #         ).cuda()
        #     )
        # model.append(
        #     PointnetSAModule(
        #         npoint=128,
        #         radius=0.4,
        #         nsample=64,
        #         mlp=[128, 128, 256],
        #         # mlp=[128, 128, 128, 256],
        #         use_xyz=True,
        #         use_APP=True,
        #     ).cuda()
        # )

        # xyz = torch.randn(B, N, 3).cuda()
        # features = torch.randn(B, 3, N).cuda()
        # for module in model:
        #     xyz, features = module(xyz, features)

        
        
        # endregion pointnet debug



    print("End debug")

    # end{lcx}
