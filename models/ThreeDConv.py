import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.threed_conv_classes import *
from models.metrics import *
from models.logging_utils import *

class ThreeDConv(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()

        self.save_hyperparameters()
        # if mod == "skip":
        #     self.mod = ThreeDConvWideFourDeepThreeSkipRGB() 
        # elif mod == "autoreg":
        #     self.mod = ThreeDConvWideFourDeepThreeSkipAutoregRGB()
        # else:
        #     raise ValueError("mod must either be 'skip' or 'autoreg") 

        self.mod = ThreeDConvWideFourDeepThreeSkipRGB()

        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        out = self.mod(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        return {"optimizer": optimizer, 
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss"
                }

    def training_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        outputs = self.forward(ctx_frames)
        loss = self.loss(outputs, tgt_frames)
        self.log('train_loss', loss, 
                 on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        pred_frames = self.forward(ctx_frames)
        loss = self.loss(tgt_frames, pred_frames)
        ssim = self.ssim(tgt_frames, pred_frames)
        psnr = self.psnr(tgt_frames, pred_frames)
        self.log_dict(
            {"val_loss": loss,
             "val_ssim": ssim,
             "val_psnr": psnr
            }, on_step=False, on_epoch=True, prog_bar=False)   

        return ctx_frames, tgt_frames, pred_frames

    def validation_epoch_end(self, validation_step_outputs):
        # Add plot to logger every 5 epochs
        if (self.current_epoch+1) % 5 == 0:
            # first batch in validation dataset
            batch_ctx, batch_tgt, batch_pred = validation_step_outputs[0]
            # first video
            ctx_frames = batch_ctx[0]
            tgt_frames = batch_tgt[0]
            pred_frames = batch_pred[0] # C x F x H x W

            img = make_plot_image(ctx_frames, tgt_frames,
                                    pred_frames, epoch=self.current_epoch+1)
            
            tb = self.logger.experiment
            tb.add_image("val_predictions", img, global_step=self.current_epoch)

class ThreeDConvAutoreg(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()

        self.save_hyperparameters()
        self.mod = ThreeDConvWideFourDeepThreeSkipAutoregRGB()

        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        out = self.mod(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        return {"optimizer": optimizer, 
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss"
                }

    def training_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        outputs = self.forward(ctx_frames)
        loss = self.loss(outputs, tgt_frames)
        self.log('train_loss', loss, 
                 on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        pred_frames = self.forward(ctx_frames)
        loss = self.loss(tgt_frames, pred_frames)
        ssim = self.ssim(tgt_frames, pred_frames)
        psnr = self.psnr(tgt_frames, pred_frames)
        self.log_dict(
            {"val_loss": loss,
             "val_ssim": ssim,
             "val_psnr": psnr
            }, on_step=False, on_epoch=True, prog_bar=False)   

        return ctx_frames, tgt_frames, pred_frames

    def validation_epoch_end(self, validation_step_outputs):
        # Add plot to logger every 5 epochs
        if (self.current_epoch+1) % 5 == 0:
            # first batch in validation dataset
            batch_ctx, batch_tgt, batch_pred = validation_step_outputs[0]
            # first video
            ctx_frames = batch_ctx[0]
            tgt_frames = batch_tgt[0]
            pred_frames = batch_pred[0] # C x F x H x W

            img = make_plot_image(ctx_frames, tgt_frames,
                                    pred_frames, epoch=self.current_epoch+1)
            
            tb = self.logger.experiment
            tb.add_image("val_predictions", img, global_step=self.current_epoch)
        
        
