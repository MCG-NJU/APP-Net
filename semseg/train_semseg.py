import os

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import sys
import time
import glob
import logging
sys.path.append("../")
import numpy as np
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

import torch
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)

def backup_code(dst_path):
    base_dir = os.path.dirname('../..')
    print('base_dir = ', base_dir)
    print('abs path = ', os.path.abspath(__file__))
    for pattern in ['*.py', 'semseg/config', 'semseg/models', 'semseg/data', 'pointnet2_ops_lib', 'semseg/*.py']:
        for src in glob.glob(os.path.join(base_dir, pattern)):
            if 'build' not in src:
                dst = os.path.join(dst_path, 'backup')
                logging.info('Copying %s -> %s' % (os.path.relpath(src), os.path.relpath(dst)))
                os.makedirs(dst, exist_ok=True)
                os.system("cp -r {} {}".format(src, dst))

    
    
@hydra.main("config/config_semseg.yaml")
def main(cfg):
    log_folder = os.path.join(cfg.dataset.name, cfg.task_model.name, cfg.exp_name, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(log_folder, "logs/"))
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_folder, 'train.log'), filemode='w')
    backup_code(log_folder)
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg)).cuda()
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="mIoU",
        mode="max",
        save_top_k=4,
        filepath=os.path.join(
            log_folder, "{epoch}-{mIoU:.2f}"
        ),
        verbose=True,
    )

    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        early_stop_callback=None, #early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.distrib_backend,
        logger=tb_logger,

    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
