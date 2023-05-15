# APP-Net: Auxiliary-point-based Push and Pull Operations for Efficient Point Cloud Classification [arxiv](https://arxiv.org/abs/2205.00847)
> [Tao Lu](https://github.com/inspirelt), Chunxu Liu, Youxi Chen, Gangshan Wu, [Limin Wang](http://wanglimin.github.io/)<br>Multimedia Computing Group, Nanjing University

Introduction
----
APP-Net is a fast and memory-efficient backbone for point cloud recognition. The efficiency comes from that the total computation complexity is linear to the input points. To achieve this goal, we abandon the FPS+kNN (or Ball Query) paradigm and propose a RandomSample+1NN aggregator. To the best of our knowledge, APP-Net is the first pure-cluster-based backbone for point cloud processing.

Setup
-----
The version of below dependencies can be modified according to your machine. The tested system is Ubuntu 16.04.
```
conda create -n APPNet python=3.7
conda activate APPNet
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install -e pointnet2_ops_lib/
pip install -r requirements.txt
python install python-pcl
```

Datasets
--------

### ModelNet40:

Download it from the official website and then unzip it to data folder:
```
mkdir -p classification/datasets
cd classification/datasets
wget -c https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip --no-check-certificate
unzip modelnet40_normal_resampled.zip
```

### ScanObjectNN 
```
cd classification/datasets
```
Download data from [Google Drive](https://drive.google.com/file/d/1v6-JXeBlNvTjKLNlbDKhdc3f33u8PtX7/view?usp=sharing), [Official](https://hkust-vgd.github.io/scanobjectnn/) or [BaiduYun](https://pan.baidu.com/s/1xDWOY3s9XTrv3DciJ97WiQ)(cdn4).
```
unzip scanobjectnn.zip
```

### S3DIS 

We support two versions of S3DIS, with slight difference in splitting the scenes. The first is the adopted by [PointNet2_Pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master). 
```
mkdir -p semseg/dataset
cd semseg/dataset
wget -c https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip --no-check-certificate
unzip indoor3d_sem_seg_hdf5_data.zip
```

The second is adopted by the [PointCNN](https://github.com/yangyanli/PointCNN) and [PVCNN](https://github.com/mit-han-lab/pvcnn). You can follow them to pre-process the data. We provide a processed one in [BaiduYun](https://pan.baidu.com/s/1CPFTGnJ0OuUlGpGFZhfGTw)(5t2n), which can be uncompressed with 
```
cat pointcnn.tar.gz* | tar zx
```




The dataset file structure should be like

```
classification/
               datasets/
                        modelnet40_normal_resampled_cache/
                                                          train/
                                                          test/
                        scanobjectnn/
                                    h5_files/main_split/
                                                        training_objectdataset_augmentedrot_scale75.h5
                                                        test_objectdataset_augmentedrot_scale75.h5
-----------------------------------------------------------------------------------------------------------
semseg/
      dataset/
              indoor3d_sem_seg_hdf5_data/
                                         all_files.txt
                                         room_filelist.txt
                                         ply_data_all_0.h5
                                         *.h5
              pointcnn/
                       Area_1/
                       Area_2/
                       Area_3/
                       Area_4/
                       Area_5/
                       Area_6/
                                      
              
                                     
               
```








Usage
----------------
We provide scripts for simplifying the starting process. To train classification, type 

```
sh cls_train.sh
```
 To train segmentation (coming soon), type 
 
 ```
 sh segmseg_train.sh
 ```
 
 To change the experiment dataset and backbone, modify the related keyword in
 ```
 {task}/config/config_{task}.yaml
 ```
 
 `{task}` refers to `classification` or `semseg`. 



Citation
--------

```
  @misc{lu2022appnet,
      title={APP-Net: Auxiliary-point-based Push and Pull Operations for Efficient Point Cloud Classification}, 
      author={Tao Lu and Chunxu Liu and Youxin Chen and Gangshan Wu and Limin Wang},
      year={2022},
      eprint={2205.00847},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Acknowledgement
---------------

This project is based on the following repos, sincerely thanks to the efforts:

[Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch),
[pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch),
[pvcnn](https://github.com/mit-han-lab/pvcnn.git).
