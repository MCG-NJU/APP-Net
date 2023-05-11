"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import pcl
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def load_scanobjectnn_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []
    h5_name = BASE_DIR + '/../datasets/scanobjectnn/h5_files/main_split/' + partition + f'_objectdataset_augmentedrot_scale75.h5' # _withnormaldisp_k=10.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    if 'normal' in f.keys():
        normal = f['normal'][:].astype('float32')
        data = np.concatenate([data, normal], axis=-1)
    label = f['label'][:].astype('int64')

    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

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
    rotation_angle = np.random.uniform() * 2.0 * np.pi
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

    # translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    translated_pointcloud = np.add(pointcloud, xyz2).astype('float32')
    # translated_pointcloud = pointcloud
    return translated_pointcloud

def kSearchNormalEstimation(cloud, num_neighbors):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_KSearch(num_neighbors)
    normals = ne.compute().to_array()

    return normals

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class ScanObjectNNLoader(Dataset):
    def __init__(self, input_feat, split = 'training', num_points = 1024):
        self.data, self.label = load_scanobjectnn_data(split)
        self.num_points = num_points
        self.input_feat = input_feat
        self.split = split

    def __getitem__(self, item):
        pointcloud = self.data[item][:]
        idx = np.random.choice(pointcloud.shape[0], self.num_points, replace=False)
        pointcloud = pointcloud[idx][:,:]
        label = self.label[item]
        xyz = pointcloud[:, :3]
        if self.split == 'training':
            xyz = rotate_pointcloud(xyz)
            xyz = translate_pointcloud(xyz)
        # xyz = pc_normalize(xyz)
        if 'gn' in self.input_feat:
            AssertionError('groundtruth normal is not available in ScanObjectNN!')
        elif 'n' in self.input_feat or 'c' in self.input_feat:
            pcd = pcl.PointCloud(xyz)
            normal_curvature = kSearchNormalEstimation(pcd, num_neighbors=10)
            normal = normal_curvature[:,:3]
            curvature = normal_curvature[:,3:]
            
            if 'nc' in self.input_feat:
                pointcloud = np.concatenate([xyz, normal, curvature], axis=-1)
            elif 'n' in self.input_feat:
                pointcloud = np.concatenate([xyz, normal], axis=-1)
            else:
                pointcloud = np.concatenate([xyz, curvature], axis=-1)
            
        np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN_MLP('training', 1024)
    test = ScanObjectNN_MLP('test', 1024)
    for data, label in train:
        print(data.shape)
        print(label)
