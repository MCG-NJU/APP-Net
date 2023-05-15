import pytorch_lightning as pl
import os
import sys
import time
sys.path.append("../..")
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from warmup_scheduler import GradualWarmupScheduler
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, APPblock, SharedMLP, APPFPModule
from torch.utils.data import DataLoader

from .lovasz_losses import lovasz_softmax


from data import S3DIS_pn2, S3DIS_pvcnn


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


lr_clip = 1e-5
bnm_clip = 1e-2


class APPSemSeg(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        print(f'hparams: {hparams}')
        self.dataset = hparams['dataset.name']
        self.num_points = hparams['dataset.num_points']
        self.num_classes = hparams['dataset.num_classes']
        self.input_feat = hparams['dataset.input_feat']
        self.input_dim =('p' in self.input_feat)*3+('P' in self.input_feat)*3+('r' in self.input_feat)*3+('n' in self.input_feat)*3+('c' in self.input_feat)*1
        self.down_sample_rate = [int(_) for _ in hparams["architecture.down_sample_rate"].split(',')] 
        self.auxiliary_rate = [int(_) for _ in hparams['architecture.auxiliary_rate'].split(',')]
        self.train_iter = hparams['architecture.train_iter']
        self.test_iter = hparams['architecture.test_iter']
        assert len(self.down_sample_rate) == len(self.auxiliary_rate)
        self.network_depth = len(self.down_sample_rate)
        self.cs = [int(_) for _ in hparams["architecture.channels"].split(',')]
        cs_ratio = hparams['architecture.channel_ratio']
        self.cs = [int(_ * cs_ratio) for _ in self.cs]
        self.baseop = hparams['model.op']
        self.data_loader = hparams['dataset.loader']
        self.color_drop = hparams['dataset.color_drop']
        self.lovasz_loss = lovasz_softmax
        self.smooth = True
        self.hparams = hparams

        self._build_model()

    def _build_model(self):
        self.emb = SharedMLP(self.input_dim, self.cs[0])
        
        # build encoder
        self.APP_E_0 = APPblock(self.cs[0], self.cs[1], down_sample_rate=self.down_sample_rate[0], auxiliary_rate=self.auxiliary_rate[0], baseop=self.baseop, use_feat=True)
        self.APP_E_1 = APPblock(self.cs[1], self.cs[2], down_sample_rate=self.down_sample_rate[1], auxiliary_rate=self.auxiliary_rate[1], baseop=self.baseop, use_feat=True)
        self.APP_E_2 = APPblock(self.cs[2], self.cs[3], down_sample_rate=self.down_sample_rate[2], auxiliary_rate=self.auxiliary_rate[2], baseop=self.baseop, use_feat=True)
        self.APP_E_3 = APPblock(self.cs[3], self.cs[4], down_sample_rate=self.down_sample_rate[3], auxiliary_rate=self.auxiliary_rate[3], baseop=self.baseop, use_feat=True)
        self.APP_E_4 = APPblock(self.cs[4], self.cs[5], down_sample_rate=self.down_sample_rate[4], auxiliary_rate=self.auxiliary_rate[4], baseop=self.baseop, use_feat=True)
        
        
        # build decoder
        self.APP_D_4 = APPFPModule(mlp=[self.cs[5]+self.cs[4], self.cs[4]])
        self.APP_D_3 = APPFPModule(mlp=[self.cs[4]+self.cs[3], self.cs[3]])
        self.APP_D_2 = APPFPModule(mlp=[self.cs[3]+self.cs[2], self.cs[2]])
        self.APP_D_1 = APPFPModule(mlp=[self.cs[2]+self.cs[1], 128])
        self.APP_D_0 = APPFPModule(mlp=[128+self.cs[0], 128])
        
        
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.num_classes, kernel_size=1),
        )
        
    
    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._generate_position_and_features(pointcloud)
        N = xyz.shape[1]
        
        
        # stem
        f = self.emb(features.nan_to_num()) # (B, 32, N)
        if not self.training:
            iterative = self.train_iter
        else:
            iterative = self.test_iter
        # encoder
        xyz0, f0 = self.APP_E_0(xyz, f, iterative=iterative)
        xyz1, f1 = self.APP_E_1(xyz0, f0, iterative=iterative)
        xyz2, f2 = self.APP_E_2(xyz1, f1, iterative=iterative)
        xyz3, f3 = self.APP_E_3(xyz2, f2, iterative=iterative)
        xyz4, f4 = self.APP_E_4(xyz3, f3, iterative=iterative)
        
        
        # decoder
        xyz = xyz.transpose(2, 1).contiguous()
        xyz0 = xyz0.transpose(2, 1).contiguous()
        xyz1 = xyz1.transpose(2, 1).contiguous()
        xyz2 = xyz2.transpose(2, 1).contiguous()
        xyz3 = xyz3.transpose(2, 1).contiguous()
        xyz4 = xyz4.transpose(2, 1).contiguous()
        
        new_f = self.APP_D_4(xyz3, xyz4, f3, f4)
        new_f = self.APP_D_3(xyz2, xyz3, f2, new_f)
        new_f = self.APP_D_2(xyz1, xyz2, f1, new_f)
        new_f = self.APP_D_1(xyz0, xyz1, f0, new_f)
        new_f = self.APP_D_0(xyz, xyz0, f, new_f)

        return self.fc_layer(new_f)

    def _build_dataloader(self, dset, mode):
        return DataLoader(
            dset,
            batch_size=self.hparams["batch_size"],
            shuffle=mode == "train",
            num_workers=16,
            pin_memory=False,
            drop_last=mode == "train",
        )

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["optimizer.lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["optimizer.decay_step"]
                )
            ),
            lr_clip / self.hparams["optimizer.lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["optimizer.bn_momentum"]
            * self.hparams["optimizer.bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["optimizer.decay_step"]
                )
            ),
            bnm_clip,
        )
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["optimizer.lr"],
            weight_decay= self.hparams["optimizer.weight_decay"],
        )

        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=2, after_scheduler=lr_scheduler)

        
        return [optimizer], [scheduler_warmup, bnm_scheduler]

    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    def prepare_data(self):
        if self.dataset == 's3dis':
            if self.data_loader == 'pn2':
                self.train_dset = S3DIS_pn2(self.input_feat, self.hparams['dataset.val_area'], self.num_points, train=True, color_drop=self.color_drop)
                self.val_dset = S3DIS_pn2(self.input_feat, self.hparams['dataset.val_area'], self.num_points, train=False, color_drop=self.color_drop)
            elif self.data_loader == 'pvcnn':
                self.train_dset = S3DIS_pvcnn(self.input_feat, self.hparams['dataset.val_area'], self.num_points, split='train', color_drop=self.color_drop)
                self.val_dset = S3DIS_pvcnn(self.input_feat, self.hparams['dataset.val_area'], self.num_points, split='test', color_drop=self.color_drop)
            else:
                AssertionError(f"Not Supported Loader {self.data_loader} for {self.dataset}!")
        else:
            AssertionError(f"Not Supported Dataset {self.dataset}!")

    def _generate_position_and_features(self, pc):
        pc = pc.transpose(1, 2)
        xyz = pc[:,:3, :]
        additional_channel = pc[:,3:, :] if pc.size(1) > 3 else None

        if 'p' in self.input_feat:
            features = pc.contiguous()
        else:
            features = additional_channel.contiguous()
        
        return xyz, features
    
    def training_step(self, batch, batch_idx):
        pc, labels = batch
        logits = self.forward(pc)
        
        outputs = logits.unsqueeze(dim=-1)
        targets = labels.unsqueeze(dim=1).unsqueeze(dim=-1)
        loss_lovasz = self.lovasz_loss(torch.nn.functional.softmax(outputs, dim=1), targets)
                    
        loss = F.cross_entropy(logits, labels) + 1.0*loss_lovasz
        with torch.no_grad():
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
    
    def validation_step(self, batch, batch_idx):
        pc, labels = batch
        logits = self.forward(pc)
        pred = torch.argmax(logits, dim=1)
        ### total ###
        correct_total = (pred == labels).float().sum()
        seen_total = torch.tensor(labels.shape[0] * labels.shape[1]).float().cuda()

        #### class ####
        NUM_CLASSES = 13
        correct_class = torch.zeros(self.num_classes).cuda()
        iou_deno_class = torch.zeros(self.num_classes).cuda()
        for l in range(self.num_classes):
            correct_class[l] = ((pred == l) * (labels == l)).float().sum()
            iou_deno_class[l] = ((pred == l) + (labels == l)).float().sum()

        return dict(correct_total=correct_total,
                    seen_total=seen_total,
                    correct_class=correct_class,
                    iou_deno_class=iou_deno_class)
    
    def validation_end(self, outputs):
        reduced_outputs = {}
        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).sum(dim=0)

        total_correct = reduced_outputs['correct_total']
        total_seen = reduced_outputs['seen_total']
        total_correct_class = reduced_outputs['correct_class']
        total_iou_deno_class = reduced_outputs['iou_deno_class']
        total_correct_class = total_correct_class.view(-1, self.num_classes).sum(dim=0)
        total_iou_deno_class = total_iou_deno_class.view(-1, self.num_classes).sum(dim=0)

        accuracy = total_correct.sum() / total_seen.sum()
        IoU = total_correct_class / (total_iou_deno_class + 1e-6)
        mIoU = IoU.mean()
        print('------------------------------------------')
        print(f'accuracy = {accuracy.item():.5f}')
        print(f'miou = {mIoU.item():.5f}')
        print('==========================================')
        return dict(acc=accuracy, mIoU=mIoU)
