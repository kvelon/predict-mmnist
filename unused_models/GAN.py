import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.threed_conv_classes import *
from models.metrics import *
from models.logging_utils import *

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv = nn.Sequential(
            Conv3dBlock(1, 8),
            Conv3dBlock(8, 8), Conv3dBlock(8, 8), 
            Conv3dDownsample(8),
            Conv3dBlock(16, 16), Conv3dBlock(16,16),
            Conv3dDownsample(16),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            Conv3dDownsample(32),
            ConvTranspose3dBlock(64, 32), ConvTranspose3dBlock(32, 16),
            ConvTranspose3dBlock(16, 8), ConvTranspose3dBlock(8, 4), 
        )
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):   
        x = self.conv(x)
        x_flat = x.reshape(x.shape[0], -1)
        validity = self.fc(x_flat)
        return validity

class GAN(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.generator = ThreeDConvDeepTwo()
        self.discriminator = Discriminator()
        self.learning_rate = learning_rate
        self.mse = nn.MSELoss()
        self.ssim  = SSIM()
        self.psnr = PSNR()

    def forward(self, ctx_frames):
        return self.generator(ctx_frames)

    def adversarial_loss(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), 
                                 lr=self.learning_rate,
                                 betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), 
                                 lr=self.learning_rate,
                                 betas=(0.5, 0.999))
        
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        ctx_frames, tgt_frames = batch

        ################################
        ###### Optimize generator ######
        ################################
        if optimizer_idx == 0:
            pred_frames = self(ctx_frames)

            # We want generator to generate "real" images
            real_labels = torch.ones(tgt_frames.shape[0], 1)
            real_labels = real_labels.type_as(tgt_frames)

            g_loss = self.adversarial_loss(self.discriminator(pred_frames), real_labels)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss
        
        ################################
        #### Optimize discriminator ####
        ################################
        if optimizer_idx == 1:

            real_labels = torch.ones(tgt_frames.shape[0], 1)
            real_labels = real_labels.type_as(tgt_frames)
            real_loss = self.adversarial_loss(self.discriminator(tgt_frames), real_labels)

            fake_labels = torch.zeros(tgt_frames.shape[0], 1)
            fake_labels = real_labels.type_as(tgt_frames)
            fake_loss = self.adversarial_loss(self.discriminator(self(ctx_frames).detach()), fake_labels)
            
            d_loss = (real_loss + fake_loss) / 2  # Discriminator loss is average of two losses
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        pred_frames = self(ctx_frames)
        mse = self.mse(tgt_frames, pred_frames)
        ssim = self.ssim(tgt_frames, pred_frames)
        psnr = self.psnr(tgt_frames, pred_frames)
        self.log_dict(
            {"val_mse": mse,
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