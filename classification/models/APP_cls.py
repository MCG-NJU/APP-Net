from ast import Assert
import pytorch_lightning as pl

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from warmup_scheduler import GradualWarmupScheduler
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, APPblock, SharedMLP # , SparseAPPblock
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

sys.path.append('..')
import data.data_utils as d_utils
from data.ModelNet40Loader import ModelNet40Cls
from data.ScanObjectNNLoader import ScanObjectNNLoader


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

class APPClassification(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        print(f'hparams: {hparams}')
        self.dataset = hparams['dataset.name']
        self.num_points = hparams['dataset.num_points']
        self.num_classes = hparams['dataset.num_classes']
        self.input_feat = hparams['dataset.input_feat']
        self.input_dim =('p' in self.input_feat)*3+('P' in self.input_feat)*3+('r' in self.input_feat)*3+('n' in self.input_feat)*3+('c' in self.input_feat)*1
        self.down_sample_rate = [int(_) for _ in hparams["architecture.down_sample_rate"].split(',')] # [8, 8, 8] # baseline:[8,8,8]
        self.auxiliary_rate = [int(_) for _ in hparams['architecture.auxiliary_rate'].split(',')] # [64, 64, 64]# baseline: [64, 64, 64]
        self.train_iter = hparams['architecture.train_iter']
        self.test_iter = hparams['architecture.test_iter']
        assert len(self.down_sample_rate) == len(self.auxiliary_rate)
        self.network_depth = len(self.down_sample_rate)
        self.cs = [int(_) for _ in hparams["architecture.channels"].split(',')]
        cs_ratio = hparams['architecture.channel_ratio']
        self.cs = [int(_ * cs_ratio) for _ in self.cs]
        self.baseop = hparams['model.op']
        self.smooth = True
        self.hparams = hparams
        
        self._build_model()

    def _build_dataloader(self, dset, mode):
        return DataLoader(
            dset,
            batch_size=self.hparams["batch_size"],
            shuffle=mode == "train",
            num_workers=8,
            pin_memory=False,
            drop_last=mode == "train",
        )

    def prepare_data(self):
        if self.dataset == 'modelnet40':
            self.train_dset = ModelNet40Cls(
                self.num_points, self.input_feat, transforms=None, train=True
            )
            self.val_dset = ModelNet40Cls(
                self.num_points, self.input_feat, transforms=None, train=False
            )
        elif self.dataset == 'scanobjectnn':
            self.train_dset = ScanObjectNNLoader(self.input_feat, 'training', self.num_points)
            self.val_dset = ScanObjectNNLoader(self.input_feat, 'test', self.num_points)
        else:
            AssertionError(f'No dataset: {self.dataset}')
        
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

    def _build_model(self):
        
        self.emb = nn.Sequential(
            SharedMLP(self.input_dim, self.cs[0]),
        )

        self.APP = nn.ModuleList()
        for i in range(self.network_depth):
            self.APP.append(APPblock(self.cs[i], self.cs[i+1], down_sample_rate=self.down_sample_rate[i], auxiliary_rate=self.auxiliary_rate[i], baseop=self.baseop))
        # self.APP_1 = APPblock(self.cs[0], self.cs[1], down_sample_rate=self.down_sample_rate[0], auxiliary_rate=self.auxiliary_rate[0], baseop=self.baseop, use_feat=False)
        # self.APP_2 = APPblock(self.cs[1], self.cs[2], down_sample_rate=self.down_sample_rate[1], auxiliary_rate=self.auxiliary_rate[1], baseop=self.baseop, use_feat=False)
        # self.APP_3 = APPblock(self.cs[2], self.cs[3], down_sample_rate=self.down_sample_rate[2], auxiliary_rate=self.auxiliary_rate[2], baseop=self.baseop, use_feat=False)
        # self.APP_4 = APPblock(self.cs[3], self.cs[4], down_sample_rate=self.down_sample_rate[3], auxiliary_rate=self.auxiliary_rate[3], baseop=self.baseop, use_feat=False)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.cs[-1]+self.cs[-1], 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes),
        )

    def _generate_position_and_features(self, pc):
        pc = pc.transpose(1, 2)
        xyz = pc[:,:3, :]
        additional_channel = pc[:,3:, :] if pc.size(1) > 3 else None

        if 'p' in self.input_feat and ('n' in self.input_feat or 'c' in self.input_feat):
            features = pc.contiguous()
        elif 'p' in self.input_feat:
            features = xyz.contiguous()
        elif 'n' in self.input_feat or 'c' in self.input_feat:
            features = additional_channel.contiguous()
        else:
            AssertionError('Unsupported input feature type!')
        return xyz.contiguous(), features.contiguous()

    def forward(self, pointcloud, mode='test'):
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
        f = self.emb(features)
        
        
        if not self.training:
            iterative = self.train_iter
        else:
            iterative = self.test_iter
        
        for i in range(self.network_depth):
            xyz, f = self.APP[i](xyz, f, iterative=iterative)
        
        f_out = f
        f_max = F.max_pool2d(
            f_out, kernel_size=[1, f_out.size(2)]
        ).squeeze(-1)  # (B, C1)
        f_mean = F.avg_pool2d(
            f_out, kernel_size=[1, f_out.size(2)]
        ).squeeze(-1)  # (B, C1)

        f_final = torch.cat([f_max, f_mean], dim=1)
        if mode=='retr':
            return f_final, self.fc_layer(f_final)    
        return self.fc_layer(f_final)#, f_final


    def training_step(self, batch, batch_idx):
        pc, labels = batch
        logits = self.forward(pc, mode='train')
        
        #### smooth loss ####
        gold = labels.contiguous().view(-1)
        pred = logits
        if self.smooth:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')
        #### smooth loss end ####

        with torch.no_grad():
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
            
    def validation_step(self, batch, batch_idx):
        pc, labels = batch
        logits = self.forward(pc)

        loss = F.cross_entropy(logits, labels)
        acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        return dict(val_loss=loss, val_acc=acc)

    def validation_end(self, outputs):
        reduced_outputs = {}
        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs