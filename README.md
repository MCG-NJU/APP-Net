# APP-Net: Auxiliary-point-based Push and Pull Operations for Efficient Point Cloud Classification [arxiv](https://arxiv.org/abs/2205.00847)
> [Tao Lu](https://github.com/inspirelt), Chunxu Liu, Youxi Chen, Gangshan Wu, [Limin Wang](http://wanglimin.github.io/)<br>Multimedia Computing Group, Nanjing University

Introduction
----
APP-Net is a fast and memory-efficient backbone for point cloud recognition. The efficiency comes from that the total computation complexity is linear to the input points. To achieve this goal, we abandon the FPS+kNN (or Ball Query) paradigm and propose a RandomSample+1NN aggregator. To the best of our knowledge, APP-Net is the first pure-cluster-based backbone for point cloud processing.

Setup
-----

* Install ``python`` -- This repo is tested with ``{3.7}``

* Install ``pytorch`` with CUDA -- This repo is tested with ``{1.9}``.
  >Other versions of ``python`` and ``pytorch`` should also work, feel free to try, as along as there exists compatible ``pytorch_scatter``, but this is not guaranteed.


* Install dependencies
  ```
  pip install numpy, msgpack-numpy, lmdb, h5py, hydra-core==0.11.3, pytorch-lightning==0.7.1
  conda install pytorch-scatter -c pyg
  pip install ./pointnet2_ops_lib/.
  ```
  
Datasets
--------

### ModelNet40:

Download it from the official website and then unzip it to data folder:
```
mkdir -p classification/datasets
cd classification/datasets
wget -c https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
unzip modelnet40_normal_resampled.zip
```

### ScanObjectNN
Silimar steps as ModelNet40:
```
mkdir -p classification/datasets/scanobjectnn
cd classification/datasets
wget -c http://103.24.77.34/scanobjectnn/h5_files.zip
unzip h5_files.zip
```

The final file structure for classification should be like

```
classification/
               datasets/
                        modelnet40_normal_resampled_cache/
                        scanobjectnn/
               
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
[pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch)
