from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union
import torch.distributed as dist
import random
import tqdm
import pandas as pd

import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CyclicLR, LambdaLR, CosineAnnealingLR

from module.feature import Mel_Spectrogram
from module.loader import SPK_datamodule
import score as score
from loss import softmax, amsoftmax, aamsoftmax

import torchaudio
import torchaudio.compliance.kaldi as kaldi

class Task(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        batch_size: int = 32,
        num_workers: int = 10,
        max_epochs: int = 1000,
        trial_path: str = "data/vox1_test.txt",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.trials = np.loadtxt(self.hparams.trial_path, str)
        self.mel_trans = Mel_Spectrogram()

        from module.resnet import resnet34, resnet18, resnet34_large
        from module.ecapa_tdnn import ecapa_tdnn, ecapa_tdnn_large
        from module.conformer import conformer
        from module.conformer_cat import conformer_cat

        if self.hparams.encoder_name == "resnet18":
            self.encoder = resnet18(embedding_dim=self.hparams.embedding_dim, pooling_type=self.hparams.pooling_type)

        elif self.hparams.encoder_name == "resnet34":
            self.encoder = resnet34(embedding_dim=self.hparams.embedding_dim, pooling_type=self.hparams.pooling_type)
        
        elif self.hparams.encoder_name == "resnet34_large":
            self.encoder = resnet34_large(embedding_dim=self.hparams.embedding_dim, pooling_type=self.hparams.pooling_type)

        elif self.hparams.encoder_name == "ecapa_tdnn":
            self.encoder = ecapa_tdnn(embedding_dim=self.hparams.embedding_dim, pooling_type=self.hparams.pooling_type)

        elif self.hparams.encoder_name == "ecapa_tdnn_large":
            self.encoder = ecapa_tdnn_large(embedding_dim=self.hparams.embedding_dim, pooling_type=self.hparams.pooling_type)
        
        elif self.hparams.encoder_name == "conformer":
            print("conformer num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = conformer(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer,
                    pos_enc_layer_type=self.hparams.pos_enc_layer_type)

        elif self.hparams.encoder_name == "conformer_cat":
            print("conformer_cat num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = conformer_cat(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer,
                    pos_enc_layer_type=self.hparams.pos_enc_layer_type)

        else:
            raise ValueError("encoder name error")

        if self.hparams.loss_name == "amsoftmax":
            self.loss_fun = aamsoftmax(**dict(self.hparams))
        elif self.hparams.loss_name == "aamsoftmax":
            self.loss_fun = aamsoftmax(**dict(self.hparams))
        else:
            self.loss_fun = softmax(**dict(self.hparams))
        
        self.start_epoch = self.hparams.start_epoch
        self.my_tensor = torch.randn(3, 4)

    def forward(self, x):
        #feature = self.mel_trans(x)
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        feature, label = batch
        #feature1 = self.mel_trans(waveform)
        embedding = self(feature)
        loss, acc = self.loss_fun(embedding, label)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return loss
    
    def on_train_epoch_start(self):
        if self.current_epoch < self.start_epoch:
            for name, param in self.encoder.conformer.named_parameters():
                if name != 'after_norm.weight' and name != 'after_norm.bias':
                    param.requires_grad_(False)
        else:
            if self.hparams.encoder_name == "conformer" or self.hparams.encoder_name == "conformer_cat":
                for param in self.encoder.conformer.parameters():
                    param.requires_grad_(True)

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def on_validation_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, path = batch
        path = path[0]
        with torch.no_grad():
            #x = self.mel_trans(x)
            self.encoder.eval()
            x = self.encoder(x) #[1, 256]
            #x = F.normalize(x, p=2, dim=1)
        x = x.detach().cpu().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[path] = batch_idx

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        num_gpus = torch.cuda.device_count()
        eval_vectors = [None for _ in range(num_gpus)]
        dist.all_gather_object(eval_vectors, self.eval_vectors)
        eval_vectors = np.vstack(eval_vectors) # (4078, 256)

        table = [None for _ in range(num_gpus)]
        dist.all_gather_object(table, self.index_mapping)

        index_mapping = {}
        for i in table:
            index_mapping.update(i)

        if self.hparams.asnorm:
            self.cohort_path = np.loadtxt(self.hparams.cohort_path, str) 
            labels, scores = score.cosine_score_asnorm(self.trials, index_mapping, eval_vectors, self.cohort_path, self.hparams.topk)
        else:
            labels, scores = score.cosine_score(self.trials, index_mapping, eval_vectors)
        
        EER, threshold = score.compute_eer(labels, scores)
        print("\ncosine EER: {:.3f}% with threshold {:.3f}".format(EER*100, threshold))
        self.log("cosine_eer", EER*100)

        minDCF, threshold = score.compute_minDCF(labels, scores, p_target=0.01)
        print("cosine minDCF(10-2): {:.5f} with threshold {:.3f}".format(minDCF, threshold))
        self.log("cosine_minDCF(10-2)", minDCF)

        minDCF, threshold = score.compute_minDCF(labels, scores, p_target=0.001)
        print("cosine minDCF(10-3): {:.5f} with threshold {:.3f}".format(minDCF, threshold))
        self.log("cosine_minDCF(10-3)", minDCF)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        #scheduler = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up learning_rate
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step +
                           1) / float(self.hparams.warmup_step))
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.learning_rate
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        (args, _) = parser.parse_known_args()

        parser.add_argument("--num_workers", default=40, type=int)
        parser.add_argument("--embedding_dim", default=256, type=int)
        parser.add_argument("--num_classes", type=int, default=1211)

        parser.add_argument("--second", type=int, default=3)
        parser.add_argument('--step_size', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument("--batch_size", type=int, default=80)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--warmup_step", type=float, default=4000)
        parser.add_argument("--weight_decay", type=float, default=0.000001)

        parser.add_argument("--save_dir", type=str, default=None)
        parser.add_argument("--checkpoint_path", type=str, default=None)
        parser.add_argument("--loss_name", type=str, default="amsoftmax")
        parser.add_argument("--encoder_name", type=str, default="resnet34")
        parser.add_argument("--pooling_type", type=str, default="ASP")
        
        parser.add_argument("--num_blocks", type=int, default=6)
        parser.add_argument("--input_layer", type=str, default="conv2d")
        parser.add_argument("--pos_enc_layer_type", type=str, default="abs_pos")


        parser.add_argument("--train_csv_path", type=str, default="data/train.csv")
        parser.add_argument("--trial_path", type=str, default="data/vox1_test.txt")
        parser.add_argument("--score_save_path", type=str, default=None)

        parser.add_argument('--eval', action='store_true')

        parser.add_argument('--speed_perturb_flag', action='store_true')
        parser.add_argument('--add_reverb_noise', action='store_true') 
        parser.add_argument('--noise_csv_path', type=str, default="data/musan_lst.csv")
        parser.add_argument('--rir_csv_path', type=str, default="data/rirs_lst.csv")
        parser.add_argument('--spec_aug_flag', action='store_true')
        
        # loss functions
        parser.add_argument('--margin', type=float, default=0.2)
        parser.add_argument('--scale', type=float, default=30)

        parser.add_argument("--pre_asr_path", type=str, default=None)

        parser.add_argument("--do_lm_path", type=str, default=None)

        parser.add_argument("--start_epoch", type=int, default=0)

        parser.add_argument('--asnorm', action='store_true')
        parser.add_argument("--topk", type=int, default=300)
        parser.add_argument('--cohort_path', type=str, default="data/cohort.txt")

        return parser


def cli_main():
    seed_everything(42, workers=True)

    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    
    parser = Task.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Task(**args.__dict__)

    model_dict = model.state_dict()

    if args.pre_asr_path is not None:
        print(f'add asr model from {args.pre_asr_path}')
        state_dict = torch.load(args.pre_asr_path, map_location="cpu")
        pretrained_dict = {}
        for k,v in state_dict.items():
            k = k.split(".")
            k.insert(1,'conformer')
            k = ".".join(k)
            if k in model_dict:
                pretrained_dict[k] = v
        del pretrained_dict['encoder.conformer.after_norm.weight']
        del pretrained_dict['encoder.conformer.after_norm.bias']
        #print(pretrained_dict.keys())
        #print(model.state_dict())
        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)
    
    # Large margin fine-tuning
    if args.do_lm_path is not None:
        print("load weight from {}".format(args.do_lm_path))
        state_dict = torch.load(args.do_lm_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    
    assert args.save_dir is not None
    checkpoint_callback = ModelCheckpoint(monitor='cosine_eer', save_top_k=100,
           filename="{epoch}_{cosine_eer:.3f}", dirpath=args.save_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir='logs/', name='my_model')
    dm = SPK_datamodule(train_csv_path=args.train_csv_path, trial_path=args.trial_path, second=args.second,
            batch_size=args.batch_size, num_workers=args.num_workers, num_classes=args.num_classes,
            speed_perturb_flag = args.speed_perturb_flag,
            add_reverb_noise = args.add_reverb_noise,
            spec_aug_flag = args.spec_aug_flag,
            asnorm = args.asnorm,
            cohort_path = args.cohort_path)
    AVAIL_GPUS = torch.cuda.device_count()
    trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=AVAIL_GPUS,
            strategy="ddp",
            num_sanity_val_steps=-1,
            sync_batchnorm=True,
            callbacks=[checkpoint_callback, lr_monitor],
            default_root_dir=args.save_dir,
            reload_dataloaders_every_n_epochs=1,
            accumulate_grad_batches=1,
            logger=logger,
            )
    if args.eval:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        print("load weight from {}".format(args.checkpoint_path))
        trainer.test(model, datamodule=dm)
    else:
        if args.checkpoint_path is not None:
            trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoint_path)
        else:
            trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()

