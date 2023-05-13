import os
import sys
import time
from turtle import color
import pcl
import h5py
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from prefetch_generator import background


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
from .scannet_config import CONF

def kSearchNormalEstimation(cloud, num_neighbors):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_KSearch(num_neighbors)
    normals = ne.compute().to_array()
    
    return normals

class ScannetDataset():
    def __init__(self, input_feat, npoints=8192, phase='train', color_drop=0.4, is_weighting=True):
        self.phase = phase
        assert phase in ["train", "val", "test"]
        self.input_feat = input_feat
        if phase == 'train':
            self.scene_list = self._get_scene_list(CONF.SCANNETV2_TRAIN)
        elif self.phase == 'val':
            self.scene_list = self._get_scene_list(CONF.SCANNETV2_VAL)
        self.num_classes = 21
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.color_drop = color_drop
        
        print(f'self.scene_list: {len(self.scene_list)}')
        
        self.chunk_data = {} # init in generate_chunks()
        self._prepare_weights()
        # self.generate_chunks()
        

    def _prepare_weights(self):
        self.scene_data = {}
        scene_points_list = []
        semantic_labels_list = []
        
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            label = scene_data[:, 10]

            # append
            scene_points_list.append(scene_data)
            semantic_labels_list.append(label)
            self.scene_data[scene_id] = scene_data


        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)

    def _get_scene_list(self, path):
        scene_list = []
        with open(path) as f:
            for scene_id in f.readlines():
                scene_list.append(scene_id.strip())
        
        return scene_list

    # @background()
    def __getitem__(self, index):
        index = index % len(self.scene_list)
        # load chunks
        scene_id = self.scene_list[index]
        scene = self.scene_data[scene_id]
        
        semantic = scene[:, 10].astype(np.int32)
            
        coordmax = np.max(scene, axis=0)[:3]
        coordmin = np.min(scene, axis=0)[:3]
        
        for _ in range(5):
            curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
            curmin = curcenter-[0.75,0.75,1.5]
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
            cur_point_set = scene[curchoice]
            cur_semantic_seg = semantic[curchoice]
            
            if len(cur_semantic_seg)==0:
                continue

            mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

            if isvalid:
                chunk = cur_point_set

                choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
                chunk = chunk[choices]
                
                break
            
            # store chunk
            
            chunk = cur_point_set

            choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
            chunk = chunk[choices]
            
            
        # unpack
        point_set = chunk[:, :3] # include xyz by default
        rgb = chunk[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = chunk[:, 6:9]
        label = chunk[:, 10].astype(np.int32)
        

        if 'r' in self.input_feat:
            point_set = np.concatenate([point_set, rgb], axis=-1)
        if 'gn' in self.input_feat:
            point_set = np.concatenate([point_set, normal], axis=-1)
        if 'P' in self.input_feat:
            AssertionError('global location is not needed in ScanNetv2!')
        # if ('n' in self.input_feat and 'gn' not in self.input_feat) or 'c' in self.input_feat:
        #     pcd = pcl.PointCloud(point_set[:,:3])
        #     normal_curvature = kSearchNormalEstimation(pcd, num_neighbors=10)
        #     normal = normal_curvature[:,:3]
        #     curvature = normal_curvature[:,3:]
            
        #     if 'n' in self.input_feat:
        #         point_set = np.concatenate([point_set, normal], axis=-1)
        #     if 'c' in self.input_feat:
        #         point_set = np.concatenate([point_set, curvature], axis=-1)
        

        if self.phase == "train":
            point_set = self._augment(point_set)
        
        # prepare mask
        curmin = np.min(point_set, axis=0)[:3]
        curmax = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (curmin - 0.01)) * (point_set[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask


        point_set = torch.from_numpy(point_set).float()
        label = torch.from_numpy(label).long()

        return point_set, label # , sample_weight, fetch_time

    def __len__(self):
        count = 0
        for k in self.scene_data.keys():
            count += self.scene_data[k].shape[0] // self.npoints
        return count # len(self.scene_points_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords

        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]

        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunks(self):
        """
            note: must be called before training
        """

        print("generate new chunks for {}...".format(self.phase))
        for scene_id in tqdm(self.scene_list):
            scene = self.scene_data[scene_id]
            semantic = scene[:, 10].astype(np.int32)
            
            coordmax = np.max(scene, axis=0)[:3]
            coordmin = np.min(scene, axis=0)[:3]
            
            for _ in range(5):
                curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
                curmin = curcenter-[0.75,0.75,1.5]
                curmax = curcenter+[0.75,0.75,1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
                cur_point_set = scene[curchoice]
                cur_semantic_seg = semantic[curchoice]
                
                if len(cur_semantic_seg)==0:
                    continue

                mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
                vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

                if isvalid:
                    break
            
            # store chunk
            
            chunk = cur_point_set

            choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
            chunk = chunk[choices]
            self.chunk_data[scene_id] = chunk
            
        print("done!\n")

class ScannetDatasetWholeScene():
    def __init__(self, scene_list, npoints=8192, is_weighting=True, use_color=False, use_normal=False, use_multiview=False):
        self.scene_list = scene_list
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview

        self._load_scene_file()

    def _load_scene_file(self):
        self.scene_points_list = []
        self.semantic_labels_list = []
        
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            label = scene_data[:, 10].astype(np.int32)
            self.scene_points_list.append(scene_data)
            self.semantic_labels_list.append(label)

            if self.use_multiview:
                feature = multiview_database.get(scene_id)[()]
                self.multiview_data.append(feature)

        if self.is_weighting:
            labelweights = np.zeros(CONF.NUM_CLASSES)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(CONF.NUM_CLASSES + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(CONF.NUM_CLASSES)

    @background()
    def __getitem__(self, index):
        start = time.time()
        scene_data = self.scene_points_list[index]

        # unpack
        point_set_ini = scene_data[:, :3] # include xyz by default
        color = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]

        if self.use_color:
            point_set_ini = np.concatenate([point_set_ini, color], axis=1)

        if self.use_normal:
            point_set_ini = np.concatenate([point_set_ini, normal], axis=1)

        if self.use_multiview:
            multiview_features = self.multiview_data[index]
            point_set_ini = np.concatenate([point_set_ini, multiview_features], axis=1)

        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = point_set_ini[:, :3].max(axis=0)
        coordmin = point_set_ini[:, :3].min(axis=0)
        xlength = 1.5
        ylength = 1.5
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*xlength, j*ylength, 0]
                curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                mask = np.sum((point_set_ini[:, :3]>=(curmin-0.01))*(point_set_ini[:, :3]<=(curmax+0.01)), axis=1)==3
                cur_point_set = point_set_ini[mask,:]
                cur_semantic_seg = semantic_seg_ini[mask]
                if len(cur_semantic_seg) == 0:
                    continue

                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice,:] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                # if sum(mask)/float(len(mask))<0.01:
                #     continue

                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN

        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)

        fetch_time = time.time() - start

        return point_sets, semantic_segs, sample_weights, fetch_time

    def __len__(self):
        return len(self.scene_points_list)

def collate_random(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, N, 3)
        feats                # torch.FloatTensor(B, N, 3)
        semantic_segs        # torch.FloatTensor(B, N)
        sample_weights       # torch.FloatTensor(B, N)
        fetch_time           # float
    '''

    # load data
    (
        point_set, 
        semantic_seg, 
        sample_weight,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_set = torch.FloatTensor(point_set)
    semantic_seg = torch.LongTensor(semantic_seg)
    sample_weight = torch.FloatTensor(sample_weight)

    # split points to coords and feats
    coords = point_set[:, :, :3]
    feats = point_set[:, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_seg,      # (B, N)
        sample_weight,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch

def collate_wholescene(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, C, N, 3)
        feats                # torch.FloatTensor(B, C, N, 3)
        semantic_segs        # torch.FloatTensor(B, C, N)
        sample_weights       # torch.FloatTensor(B, C, N)
        fetch_time           # float
    '''

    # load data
    (
        point_sets, 
        semantic_segs, 
        sample_weights,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_sets = torch.FloatTensor(point_sets)
    semantic_segs = torch.LongTensor(semantic_segs)
    sample_weights = torch.FloatTensor(sample_weights)

    # split points to coords and feats
    coords = point_sets[:, :, :, :3]
    feats = point_sets[:, :, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_segs,      # (B, N)
        sample_weights,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch
