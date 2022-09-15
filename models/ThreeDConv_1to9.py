import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.threed_conv_classes import *
from models.metrics import *
from models.logging_utils import *

class ThreeDConv_1to9(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.mod = ThreeDConvWideFourDeepThreeSkipRGB()
        self.loss = nn.MSELoss()
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
        ctx_frame, tgt_frames = batch
        num_tgt_frames = tgt_frames.shape[2]
        losses = torch.zeros(num_tgt_frames, dtype=torch.float32)

        for i in range(num_tgt_frames):
            output = self.forward(ctx_frame)
            losses[i] = self.loss(output, tgt_frames[:, :, i:i+1])
            ctx_frame = output

        self.log('train_loss', losses.mean(), 
                    on_step=False, on_epoch=True, prog_bar=True)

        return losses.mean()

    def validation_step(self, batch, batch_idx):
        ctx_frame, tgt_frames = batch
        num_tgt_frames = tgt_frames.shape[2]
        losses = torch.zeros(num_tgt_frames, dtype=torch.float32)

        pred_frames = torch.zeros_like(tgt_frames)
        current_frame = ctx_frame

        for i in range(num_tgt_frames):
            output = self.forward(current_frame)
            losses[i] = self.loss(output, tgt_frames[:, :, i:i+1])
            pred_frames[:, :, i] = output.squeeze()
            current_frame = output

        ssim = self.ssim(tgt_frames, pred_frames)
        psnr = self.psnr(tgt_frames, pred_frames)

        self.log_dict(
            {"val_loss": losses.mean(),
             "val_ssim": ssim,
             "val_psnr": psnr
            }, on_step=False, on_epoch=True, prog_bar=False)   

        return ctx_frame, tgt_frames, pred_frames

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

        
        
