import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *

try:
    import pointnet2_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    # print(f'_ext_sources = {_ext_sources}')
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))
    # print(f'_ext_headers = {_ext_headers}')

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )


class Voxel_coords(Function):
    # def __init__(self, resolution, normalize=True, eps=0):
    #     super().__init__()
    #     self.r = int(resolution)
    #     self.eps = eps
    @staticmethod
    def forward(self, coords, resolution):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0) + 0.5
        norm_coords = torch.clamp(norm_coords * resolution, 0, resolution - 1)
        vox_coords_round = torch.round(norm_coords).to(torch.int32)

        return norm_coords, vox_coords_round
voxel_coords = Voxel_coords.apply
    # def extra_repr(self):
    #     return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')


class AvgVoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape
        out, indices, counts = _ext.avg_voxelize(features, coords, resolution)
        # np.savetxt('counts.txt', counts.detach().cpu().numpy())
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, resolution, resolution, resolution)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = _ext.avg_voxelize_grad(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None

avg_voxelize = AvgVoxelization.apply

class NearestDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param coords: the coordinates of points, IntTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs = _ext.nearest_devoxelize(resolution, coords, features)
        ctx.save_for_backward(coords)
        ctx.r = resolution


        return outs[0]

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        coords = ctx.saved_tensors[0]
        grad_inputs = _ext.nearest_devoxelize_grad(grad_output.contiguous(), coords, ctx.r)

        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None

nearest_devoxelize = NearestDevoxelization.apply

# class Voxelization(Function):
#     # def __init__(self, resolution):
#     #     super().__init__()
#     #     self.r = int(resolution)
#     @staticmethod
#     def forward(self, features, vox_coords, resolution):
#         return _ext.avg_voxelize(features, vox_coords, resolution)

#         # return norm_coords, vox_coords

#     # def extra_repr(self):
#     #     return 'resolution={}'.format(self.r)
# voxelization = Voxelization.apply

'''
class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = _ext.furthest_point_sampling(xyz, npoint)
        # out = torch.IntTensor(xyz.shape[0], npoint).cuda()

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply

'''

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply
class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply

class OneNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 1) l2 distance to one nearest neighbors
        idx : torch.Tensor
            (B, n, 1) index of 1 nearest neighbors
        """
        dist2, idx = _ext.one_nn(unknown, known)
        # print(f'idx0: {idx.size()}')
        # dist2 = dist2[:, :, :1].contiguous()
        # idx = idx[:, :, :1].contiguous()
        dist = torch.sqrt(dist2)
        # print(f'idx1: {idx.size()}')
        ctx.mark_non_differentiable(dist, idx)
        # print(f'idx2: {idx.size()}')
        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()

one_nn = OneNN.apply

class OneNN_pf(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
            Find the nearest neighbors of unknown in known using location and features
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, c) tensor of known features
        known : torch.Tensor
            (B, m, c) tensor of unknown features

        Returns
        -------
        idx : torch.Tensor
            (B, n, 1) index of 1 nearest neighbors
        """
        idx = _ext.one_nn_pf(unknown, known)
        ctx.mark_non_differentiable(idx)
        
        return idx

    @staticmethod
    def backward(ctx, grad_idx):
        return ()

one_nn_pf = OneNN_pf.apply

class ChamferDistance(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        r"""
            calculate chamfer_distance and return the one nn idx
        Parameters
        ----------
        xyz1 : torch.Tensor
            (B, n, 3) tensor of known features
        xyz2 : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        idx : torch.Tensor
            (B, n, 1) index of 1 nearest neighbors
        """
        batch_size, n_points_1, n_points_2 = xyz1.shape[0], xyz1.shape[1], xyz2.shape[1]

        dists1 = torch.zeros([batch_size, n_points_1], device=xyz1.device)
        idx = torch.zeros([batch_size, n_points_1], dtype=torch.int, device=xyz1.device)

        _ext.chamfer_distance(xyz1, xyz2, dists1, idx)
        idx.unsqueeze(-1)
        ctx.mark_non_differentiable(idx)

        return idx
    
    @staticmethod
    def backward(ctx, grad_idx):
        return ()

chamfer_distance = ChamferDistance.apply

class ChamferDistanceRadius(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, radius):
        r"""
            calculate chamfer_distance and return the one nn idx
        Parameters
        ----------
        xyz1 : torch.Tensor
            (B, n, 3) tensor of known features
        xyz2 : torch.Tensor
            (B, m, 3) tensor of unknown features
        radius: Scalar, radius for neighbor query
            

        Returns
        -------
        idx : torch.Tensor
            (B, n, 1) index of 1 nearest neighbors
        """
        batch_size, n_points_1, n_points_2 = xyz1.shape[0], xyz1.shape[1], xyz2.shape[1]

        dists1 = torch.zeros([batch_size, n_points_1], device=xyz1.device)
        idx = torch.zeros([batch_size, n_points_1], dtype=torch.int, device=xyz1.device)

        _ext.chamfer_distance_radius(xyz1, xyz2, radius, dists1, idx)
        idx.unsqueeze(-1)
        ctx.mark_non_differentiable(idx)

        return idx
    
    @staticmethod
    def backward(ctx, grad_idx):
        return ()

chamfer_distance_radius = ChamferDistanceRadius.apply

class OneInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type(Any, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs linear interpolation on 1 feature
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 1) one nearest neighbors of the target features in features
        
        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, features)

        return _ext.one_interpolate(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.one_interpolate_grad(
            grad_out.contiguous(), idx, m
        )

        return grad_features, torch.zeros_like(idx)

one_interpolate = OneInterpolate.apply




class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        output = _ext.ball_query(new_xyz, xyz, radius, nsample)

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


if __name__=="__main__":
    B = 1
    N = 16
    M = 4
    points = torch.rand([B, N, 3]).cuda()
    points[:,:,:2] = 0
    subpoints = furthest_point_sample(points, M)
    print(f'points = {points}')
    print(f'subpoint = {subpoints}')
    print(subpoints.shape)
