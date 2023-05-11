import os
import os.path as osp
import shlex
import shutil
import subprocess
import pcl
import time

import lmdb
import msgpack_numpy
import numpy as np

import torch
import torch.utils.data as data
import tqdm
import sys
sys.path.append('../..')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    # R = torch.from_numpy(
    #     cosval * np.eye(3)
    #     + sinval * cross_prod_mat
    #     + (1.0 - cosval) * np.outer(u, u)
    # )
    R = cosval * np.eye(3) + sinval * cross_prod_mat + (1.0 - cosval) * np.outer(u, u)
    return R

def rotate_pointcloud(points, axis=np.array([0.0, 1.0, 0.0])):
    rotation_angle = np.random.uniform() * 0.1 * np.pi
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.shape[1] > 3
    if not normals:
        return np.matmul(points, rotation_matrix.T)
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:6]
        points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
        points[:, 3:6] = np.matmul(pc_normals, rotation_matrix.T)

        return points

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    # translated_pointcloud = np.add(pointcloud, xyz2).astype('float32')
    # translated_pointcloud = pointcloud
    return translated_pointcloud

def kSearchNormalEstimation(cloud, num_neighbors):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_KSearch(num_neighbors)
    normals = ne.compute().to_array()
    return normals

class ModelNet40Cls(data.Dataset):
    def __init__(self, num_points, input_feat, transforms=None, train=True, download=True):
        super().__init__()
        # self.pre_knn = pre_knn
        # self.depth = depth
        self.input_feat = input_feat
        self.train = train
        
        # self.downsample_rate = downsample_rate
        self.transforms = transforms

        self.set_num_points(num_points)
        self._cache = os.path.join(BASE_DIR, "../datasets/modelnet40_normal_resampled_cache")
        # self._cache = "../datasets/modelnet40_normal_resampled_cache"

        if not osp.exists(self._cache):
            self.folder = "../datasets/modelnet40_normal_resampled"
            self.data_dir = os.path.join(BASE_DIR, self.folder)
            self.url = (
                "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
            )

            if download and not os.path.exists(self.data_dir):
                zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
                subprocess.check_call(
                    shlex.split("curl {} -o {} --insecure".format(self.url, zipfile))
                )

                subprocess.check_call(
                    shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
                )

                subprocess.check_call(shlex.split("rm {}".format(zipfile)))

            # self.train = train
            self.set_num_points(num_points)

            self.catfile = os.path.join(self.data_dir, "modelnet40_shape_names.txt")
            self.cat = [line.rstrip() for line in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))

            os.makedirs(self._cache)

            print("Converted to LMDB for faster dataloading while training")
            for split in ["train", "test"]:
                if split == "train":
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet40_train.txt")
                        )
                    ]
                else:
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet40_test.txt")
                        )
                    ]

                shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids]
                # list of (shape_name, shape_txt_file_path) tuple
                self.datapath = [
                    (
                        shape_names[i],
                        os.path.join(self.data_dir, shape_names[i], shape_ids[i])
                        + ".txt",
                    )
                    for i in range(len(shape_ids))
                ]

                with lmdb.open(
                    osp.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    for i in tqdm.trange(len(self.datapath)):
                        fn = self.datapath[i]
                        point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
                        cls = self.classes[self.datapath[i][0]]
                        cls = int(cls)

                        txn.put(
                            str(i).encode(),
                            msgpack_numpy.packb(
                                dict(pc=point_set, lbl=cls), use_bin_type=True
                            ),
                        )

            shutil.rmtree(self.data_dir)

        self._lmdb_file = osp.join(self._cache, "train" if train else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None

    def __getitem__(self, idx):
        t1 = time.time()
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)
        t2 = time.time()
        # print(f'load data: {t2-t1}')
        point_set = ele["pc"]
        idx = np.random.choice(ele["pc"].shape[0], self.num_points, replace=False)
        # pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(idx)

        point_set = point_set[idx, :]
        # point_set[:, :3] = pc_normalize(point_set[:,:3])
        if 'gn' in self.input_feat:
            pass
        elif 'n' in self.input_feat or 'c' in self.input_feat:
            
            # if self.input_feat['use_normal'] or self.input_feat['use_curvature']:
            xyz = point_set[:,:3]
            pcd = pcl.PointCloud(xyz)
            estimated_normal = kSearchNormalEstimation(pcd, num_neighbors=10)
            normal  = estimated_normal[:,:3]
            curvature = estimated_normal[:,3:] 
            mask = normal[:,2]<=0
            normal[mask] = -1*normal[mask]
            if 'nc' in self.input_feat:
                point_set = np.concatenate([xyz, normal, curvature], axis=-1)
            elif 'n' in self.input_feat:
                point_set = np.concatenate([xyz, normal], axis=-1)
            else:
                point_set = np.concatenate([xyz, curvature], axis=-1)
            # point_set = np.concatenate([point_set[:,:3], normal, curvature], axis=-1)

            # point_set = np.concatenate([point_set[:, :3], normal, np.abs(normal), curvature], axis=-1)
            # point_set = np.concatenate([point_set[:,:3], estimated_normal], axis=-1)
            
        if self.transforms is not None:
            # Reminder: If pre_knn is True, code won't be here.
            point_set = self.transforms(point_set)
            if torch.is_tensor(point_set):
                point_set = point_set.numpy()
         
        return point_set, ele["lbl"]

    def __len__(self):
        return self._len

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils
    from pointnet2_ops import pointnet2_utils
    import numpy as np

    def square_distance(src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def knn_point(nsample, xyz, new_xyz):
        """
        Input:
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        sqrdists = square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
        return group_idx
    
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = ModelNet40Cls(4096, train=True, pre_knn=True, depth=3, downsample_rate=4, transforms=transforms)
    print(len(dset[0][0]))
    print(len(dset[0][1][0]))
    print(dset[0][2])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=True)
    point_list, kd_1nn_list, _, _, label = dloader.__iter__().__next__()

    src, _ = _break_up_pc(point_list[0])

    res_kdtree = kd_1nn_list[0]

    centroid = src[:, :512, :]

    res_cpu = knn_point(1, centroid, src)

    src = src.cuda()
    centroid = centroid.cuda()
    print(f'src: {src.size()}, centroid: {centroid.size()}')
    _, res_cuda = pointnet2_utils.one_nn(src.contiguous(), centroid.contiguous())
    res_cuda = res_cuda.cpu()

    print(f'res_kdtree: {res_kdtree.size()}, res_cuda: {res_cuda.size()}, res_cpu: {res_cpu.size()}')

    cuda_correct = res_cuda == res_cpu
    kd_tree_correct = res_cpu == res_kdtree

    ind1 = np.where(cuda_correct==True)[0]
    ind2 = np.where(kd_tree_correct==True)[0]
    
    print(f'cuda correct: {ind1.shape}/{cuda_correct.shape}')
    print(f'kd_tree correct: {ind2.shape}/{cuda_correct.shape}')

    print(f'point_list: {src.size()}, kd_1nn_list: {kd_1nn_list[0].size()}')

    
