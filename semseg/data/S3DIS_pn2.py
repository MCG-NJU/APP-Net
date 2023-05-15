import os
import shlex
import subprocess

import sys
import h5py
import pcl
import numpy as np
import torch
import torch.utils.data as data
sys.path.append('../..')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    f = h5py.File(name, "r")
    data = f["data"][:]
    label = f["label"][:]
    return data, label


def kSearchNormalEstimation(cloud, num_neighbors):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_KSearch(num_neighbors)
    normals = ne.compute().to_array()
    
    return normals

class S3DIS_pn2(data.Dataset):
    def __init__(self, input_feat, area, num_points, train=True, color_drop=0.4, download=True, data_precent=1.0):
        super().__init__()
        self.data_precent = data_precent
        self.input_feat = input_feat
        self.color_drop = color_drop
        self.folder = "../dataset"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = (
            "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"
        )
        
        

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {} --insecure".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, self.data_dir))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train, self.num_points = train, num_points

        all_files = _get_data_files(os.path.join(self.data_dir, "indoor3d_sem_seg_hdf5_data/all_files.txt"))
        room_filelist = _get_data_files(
            os.path.join(self.data_dir, "indoor3d_sem_seg_hdf5_data/room_filelist.txt")
        )


        

        data_batchlist, label_batchlist = [], []
        for f in all_files:
            data, label = _load_data_file(os.path.join(self.data_dir, f))
            data_batchlist.append(data)
            label_batchlist.append(label)

        data_batches = np.concatenate(data_batchlist, 0)
        labels_batches = np.concatenate(label_batchlist, 0)

        test_area = f"Area_{area}"
        train_idxs, test_idxs = [], []
        for i, room_name in enumerate(room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        if self.train:
            self.points = data_batches[train_idxs, ...]
            self.labels = labels_batches[train_idxs, ...]
        else:
            self.points = data_batches[test_idxs, ...]
            self.labels = labels_batches[test_idxs, ...]

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        pointcloud = self.points[idx, pt_idxs].copy()
                
        if self.train:
            if np.random.random()<self.color_drop:
                pointcloud[:, 3:6] = 0
        
        pc = pointcloud[:,:3]

        if 'gn' in self.input_feat:
            AssertionError('groundtruth normal is not available in S3DIS!')
        if 'r' in self.input_feat:
            pc = np.concatenate([pc, pointcloud[:,3:6]], axis=-1)
            
        if 'P' in self.input_feat:
            pc = np.concatenate([pc, pointcloud[:,6:]], axis=-1)
        
        if 'n' in self.input_feat or 'c' in self.input_feat:
            pcd = pcl.PointCloud(pc[:,:3])
            normal_curvature = kSearchNormalEstimation(pcd, num_neighbors=10)
            normal = normal_curvature[:,:3]
            curvature = normal_curvature[:,3:]
            
            if 'n' in self.input_feat:
                pc = np.concatenate([pc, normal], axis=-1)
            if 'c' in self.input_feat:
                pc = np.concatenate([pc, curvature], axis=-1)
            # print(f'pc: {pc.shape}')
        current_points = torch.from_numpy(pc).float()
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()).long()

        return current_points, current_labels

    def __len__(self):
        return int(self.points.shape[0] * self.data_precent)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = Indoor3DSemSeg(16, "./", train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
