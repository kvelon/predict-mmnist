import torch
import torch.nn as nn
import pytorch_lightning as pl
from .ConvLSTM_old import ConvLSTM

from models.metrics import *
from models.logging_utils import *

# Code inspired by https://github.com/holmdk/Video-Prediction-using-PyTorch
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, h_curr, c_curr):

        combined = torch.cat([input_tensor, h_curr], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)  # Inconsistent with ConvLSTM equations on blog
        g = torch.tanh(cc_g)

        c_next = f * c_curr + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class EncoderDecoderConvLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, bias,
                 num_tgt_frames=5,
                 learning_rate=1e-3):
        super().__init__()
        
        self.mse = nn.MSELoss()
        self.ssim = SSIM()
        self.psnr = PSNR()
        self.learning_rate = learning_rate
        self.num_tgt_frames = num_tgt_frames

        self.encoder_1 = ConvLSTMCell(input_dim=input_dim,
                                      hidden_dim=hidden_dim,
                                      kernel_size=kernel_size,
                                      bias=bias)
    
        self.encoder_2 = ConvLSTMCell(input_dim=hidden_dim,
                                  hidden_dim=hidden_dim,
                                  kernel_size=kernel_size,
                                  bias=bias)

        self.decoder_1 = ConvLSTMCell(input_dim=hidden_dim,
                                  hidden_dim=hidden_dim,
                                  kernel_size=kernel_size,
                                  bias=bias)
        
        self.decoder_2 = ConvLSTMCell(input_dim=hidden_dim,
                                  hidden_dim=hidden_dim,
                                  kernel_size=kernel_size,
                                  bias=bias)
        
        self.decoder_CNN = nn.Conv3d(in_channels=hidden_dim,
                                     out_channels=output_dim,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))
        
    
    def autoencoder(self, x, num_ctx_frames, num_tgt_frames, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(num_ctx_frames):
            h_t, c_t = self.encoder_1(input_tensor=x[:, :, t],
                                      h_curr=h_t, c_curr=c_t)
            h_t2, c_t2 = self.encoder_2(input_tensor=h_t,
                                        h_curr=h_t2, c_curr=c_t2)  

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(num_tgt_frames):
            h_t3, c_t3 = self.decoder_1(input_tensor=encoder_vector,
                                        h_curr=h_t3, c_curr=c_t3) 
            h_t4, c_t4 = self.decoder_2(input_tensor=h_t3,
                                        h_curr=h_t4, c_curr=c_t4)  
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x):
        B, C, num_ctx_frames, H, W = x.shape

        # initialize hidden states
        h_t, c_t = self.encoder_1.init_hidden(batch_size=B, image_size=(H, W))
        h_t2, c_t2 = self.encoder_2.init_hidden(batch_size=B, image_size=(H, W))
        h_t3, c_t3 = self.decoder_1.init_hidden(batch_size=B, image_size=(H, W))
        h_t4, c_t4 = self.decoder_2.init_hidden(batch_size=B, image_size=(H, W))

        # autoencoder forward
        outputs = self.autoencoder(x, num_ctx_frames, self.num_tgt_frames, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        return {"optimizer": optimizer, 
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss"
                }

    def training_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch  # B x C x F x H x W
        pred_frames = self.forward(ctx_frames)
        loss = self.mse(pred_frames[:, :, 0], tgt_frames[:, :, 0])
        # loss = self.mse(pred_frames, tgt_frames)

        self.log('train_loss', loss,
                 prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        pred_frames = self.forward(ctx_frames)

        loss = self.mse(tgt_frames, pred_frames)
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