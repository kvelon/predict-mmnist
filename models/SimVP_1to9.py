import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.metrics import *
from models.logging_utils import *

class BasicConv2d(nn.Module):
    # B, C_in, H, W -> B, C_out, H', W'
    def __init__(self, in_channels, out_channels,
                  kernel_size, stride,
                  padding, transpose=False):
        super().__init__()
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=stride//2
            )
        self.norm = nn.GroupNorm(num_groups=2,
                                 num_channels=out_channels)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.nonlinearity(y)
        return y

class ConvSC(nn.Module):
    def __init__(self, in_channels, out_channels, stride, transpose=False):
        super().__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1,
                                transpose=transpose)

    def forward(self, x):
        y = self.conv(x)
        return y

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, groups):
        super().__init__()
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.nonlinearity(y)
        return y

class Inception(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 kernel_sizes=[3,5,7,11], groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, 
                              kernel_size=1, stride=1, padding=0)
        layers = []
        for size in kernel_sizes:
            layers.append(GroupConv2d(hidden_channels, out_channels, 
                                      kernel_size=size,
                                      stride=1, padding=size//2, groups=groups))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse: 
        return strides[:N][::-1]
    else: 
        return strides[:N]

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_stacks):
        super().__init__()
        strides = stride_generator(num_stacks)
        self.enc = nn.Sequential(
            ConvSC(in_channels, out_channels, strides[0]),
            *[ConvSC(out_channels, out_channels, stride=s) for s in strides[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)  
        # Skip connection; Output of first encoding layer is 
        # appended as input to decoder
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_stacks):
        super().__init__()
        strides = stride_generator(num_stacks, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(hidden_channels, hidden_channels, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*hidden_channels, hidden_channels, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
            )

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Translator(nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                 num_translator_stacks, kernel_sizes=[3, 5, 7, 11], 
                 groups=8):
        super().__init__()
        self.num_translator_stacks = num_translator_stacks

        enc_layers = [Inception(in_channels, hidden_channels//2, hidden_channels,
                                kernel_sizes=kernel_sizes, groups=groups)]
        for _ in range(1, num_translator_stacks - 1):
            enc_layers.append(Inception(hidden_channels, hidden_channels//2, 
            hidden_channels, kernel_sizes=kernel_sizes, groups=groups))
        enc_layers.append(Inception(hidden_channels, hidden_channels//2, 
            hidden_channels, kernel_sizes=kernel_sizes, groups=groups))

        dec_layers = [Inception(hidden_channels, hidden_channels//2, hidden_channels,
                                kernel_sizes=kernel_sizes, groups=groups)]
        for _ in range(1, num_translator_stacks - 1):
            dec_layers.append(Inception(2*hidden_channels, hidden_channels//2, 
            hidden_channels, kernel_sizes=kernel_sizes, groups=groups))
        dec_layers.append(Inception(2*hidden_channels, hidden_channels//2, 
            in_channels, kernel_sizes=kernel_sizes, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.reshape(B, F*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.num_translator_stacks):
            z = self.enc[i](z)
            # Save tensors for skip connection in all enc layers except the 
            # last since the last is connected to the first dec layer
            if i < self.num_translator_stacks - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.num_translator_stacks):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
        y = z.reshape(B, F, C, H, W)

        return y 

class SimVP_1to9(pl.LightningModule):
    def __init__(self, input_shape, 
                 hid_s=16, hid_t=256, 
                 N_s=4, N_t=8,
                 kernel_sizes=[3,5,7,11], 
                 groups=8,
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        C, F, H, W = input_shape
        self.enc = Encoder(C, hid_s, N_s)
        self.trans = Translator(F*hid_s, hid_t, N_t, kernel_sizes, groups)
        self.dec = Decoder(hid_s, C, N_s)

        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
        self.ssim = SSIM()
        self.psnr = PSNR()

    def forward(self, x):
        B, C, F, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # B x F x C x H x W
        x_reshaped = x.reshape(B*F, C, H, W)

        after_enc, enc1 = self.enc(x_reshaped)
        _, C_, H_, W_ = after_enc.shape

        after_enc_reshaped = after_enc.reshape(B, F, C_, H_, W_)
        after_trans = self.trans(after_enc_reshaped)
        after_trans_reshaped = after_trans.reshape(B*F, C_, H_, W_)

        output = self.dec(after_trans_reshaped, enc1)
        output = output.reshape(B, F, C, H, W)
        output = output.permute(0, 2, 1, 3, 4)
        return output

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

        self.log('train_loss', losses.mean(), on_step=False, on_epoch=True, prog_bar=True)

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