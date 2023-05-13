import os

import h5py
import pcl
import numpy as np
import torch
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# __all__ = ['S3DIS']

def kSearchNormalEstimation(cloud, num_neighbors):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_KSearch(num_neighbors)
    normals = ne.compute().to_array()

    return normals


class S3DIS_pvcnn(Dataset):
    def __init__(self, input_feat, area, num_points, split='train', color_drop=0.4, with_normalized_coords=True):
        """
        :param root: directory path to the s3dis dataset
        :param num_points: number of points to process for each scene
        :param split: 'train' or 'test'
        :param with_normalized_coords: whether include the normalized coords in features (default: True)
        :param holdout_area: which area to holdout (default: 5)
        """
        assert split in ['train', 'test']
        self.folder = '../dataset/pointcnn'
        self.root = os.path.join(BASE_DIR, self.folder)
        self.input_feat = input_feat
        self.split = split
        self.num_points = num_points
        self.holdout_area = None if area is None else int(area)
        self.color_drop = color_drop
        self.with_normalized_coords = with_normalized_coords
        # keep at most 20/30 files in memory
        self.cache_size = 20 if split == 'train' else 30
        self.cache = {}
        print('self.root = ', self.root)
        # mapping batch index to corresponding file
        areas = []
        if self.split == 'train':
            for a in range(1, 7):
                if a != self.holdout_area:
                    areas.append(os.path.join(self.root, f'Area_{a}'))
        else:
            areas.append(os.path.join(self.root, f'Area_{self.holdout_area}'))

        

        self.num_scene_windows, self.max_num_points = 0, 0
        index_to_filename, scene_list = [], {}
        filename_to_start_index = {}
        for area in areas:
            area_scenes = os.listdir(area)
            area_scenes.sort()
            for scene in area_scenes:
                current_scene = os.path.join(area, scene)
                scene_list[current_scene] = []
                for split in ['zero', 'half']:
                    current_file = os.path.join(current_scene, f'{split}_0.h5')
                    filename_to_start_index[current_file] = self.num_scene_windows
                    h5f = h5py.File(current_file, 'r')
                    num_windows = h5f['data'].shape[0]
                    self.num_scene_windows += num_windows
                    for i in range(num_windows):
                        index_to_filename.append(current_file)
                    scene_list[current_scene].append(current_file)
        self.index_to_filename = index_to_filename
        self.filename_to_start_index = filename_to_start_index
        self.scene_list = scene_list

    def __len__(self):
        return int(self.num_scene_windows)

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        if filename not in self.cache.keys():
            h5f = h5py.File(filename, 'r')
            scene_data = h5f['data']
            scene_label = h5f['label_seg']
            scene_num_points = h5f['data_num']
            if len(self.cache.keys()) < self.cache_size:
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
            else:
                victim_idx = np.random.randint(0, self.cache_size)
                cache_keys = list(self.cache.keys())
                cache_keys.sort()
                self.cache.pop(cache_keys[victim_idx])
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
        else:
            scene_data, scene_label, scene_num_points = self.cache[filename]

        internal_pos = index - self.filename_to_start_index[filename]
        current_window_data = np.array(scene_data[internal_pos]).astype(np.float32)
        current_window_label = np.array(scene_label[internal_pos]).astype(np.int64)
        current_window_num_points = scene_num_points[internal_pos]

        choices = np.random.choice(current_window_num_points, self.num_points,
                                replace=(current_window_num_points < self.num_points))
        


        pointcloud = current_window_data[choices, ...]
        label = current_window_label[choices]
        if self.split == 'train':
            if np.random.random()<self.color_drop:
                pointcloud[:,3:6] = 0

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
                # if self.split == 'train':
                #     if np.random.random()<0.5:
                #         normal *= 0
                pc = np.concatenate([pc, normal], axis=-1)
            if 'c' in self.input_feat:
                pc = np.concatenate([pc, curvature], axis=-1)
        
        current_points = torch.from_numpy(pc).float()
        current_labels = torch.from_numpy(label).long()
        
        return current_points, current_labels
        
        # else:
        #     return data[:-3, :], label


# class S3DIS(dict):
#     def __init__(self, input_feat, area, root, num_points, split=None, with_normalized_coords=True):
#         super().__init__()
#         if split is None:
#             split = ['train', 'test']
#         elif not isinstance(split, (list, tuple)):
#             split = [split]
#         for s in split:
#             self[s] = S3DISDataset(input_feat=input_feat, root=root, num_points=num_points, split=s,
#                                     with_normalized_coords=with_normalized_coords, holdout_area=area)
