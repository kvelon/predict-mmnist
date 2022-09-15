import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.metrics import *
from models.logging_utils import *

# Code inspired by https://github.com/sladewinter/ConvLSTM
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, frame_dim,
                 kernel_size, padding, activation="relu"):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.conv = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=4*hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

        self.W_i = nn.Parameter(torch.Tensor(hidden_dim, *frame_dim))
        self.W_f = nn.Parameter(torch.Tensor(hidden_dim, *frame_dim))
        self.W_o = nn.Parameter(torch.Tensor(hidden_dim, *frame_dim))

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)

        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        i_curr = torch.sigmoid(i_conv + self.W_i * c_prev)
        f_curr = torch.sigmoid(f_conv + self.W_f * c_prev)

        c_curr = (f_curr * c_prev) + (i_curr * self.activation(c_conv))
        o_curr = torch.sigmoid(o_conv + self.W_o * c_curr)
        h_curr = o_curr * self.activation(c_curr)

        return h_curr, c_curr

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, frame_dim,
                 kernel_size, padding, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.convLSTM_cell = ConvLSTMCell(input_dim, hidden_dim, frame_dim,
                                          kernel_size, padding)
        self.device = device


    def forward(self, x):
        # device = torch.device('cuda' if torch.cuda.is_available() and self.training else 'cpu')
        # print(f"ConvLSTM device: {self.device}")
        bs, c, f, h, w = x.shape

        # Space to store output
        output = torch.zeros(bs, self.hidden_dim, f, h, w, device=self.device)

        # Initialise hidden state and cell state
        hidden_state = torch.zeros(bs, self.hidden_dim, h, w, device=self.device)
        cell_state = torch.zeros(bs, self.hidden_dim, h, w, device=self.device)

        # Iterate over frames
        for time_step in range(f):

            hidden_state, cell_state = self.convLSTM_cell(x[:, :, time_step, :, :], hidden_state, cell_state)

            output[:, :, time_step, :, :] = hidden_state
        
        return output

class EncoderDecoderConvLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, frame_dim,
                 kernel_size, padding,
                 num_lstm_layers, learning_rate=1e-3, training=True):
        super().__init__()
        self.save_hyperparameters()

        self.mse = nn.MSELoss()
        self.ssim = SSIM()
        self.psnr = PSNR()
        self.learning_rate = learning_rate

        self.mod = nn.Sequential()
        #self.print_device()

        device = 'cuda' if training else 'cpu'
        self.mod.add_module(
            "convlstm1", ConvLSTM(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                frame_dim=frame_dim,
                kernel_size=kernel_size,
                padding=padding,
                device=device
            )
        )

        self.mod.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=hidden_dim)
        )

        for i in range(2, num_lstm_layers + 1):
            self.mod.add_module(
                f"convlstm{i}", ConvLSTM(
                input_dim=hidden_dim, 
                hidden_dim=hidden_dim, 
                frame_dim=frame_dim,
                kernel_size=kernel_size,
                padding=padding,
                device=device
                )
            )

            self.mod.add_module(
                f"batchnorm{i}", nn.BatchNorm3d(num_features=hidden_dim)
            )

        self.conv1 = nn.Conv2d(in_channels=hidden_dim, 
            out_channels=input_dim,
            kernel_size=kernel_size,
            padding=padding
        )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_dim, 
        #     out_channels=hidden_dim/2,
        #     kernel_size=kernel_size,
        #     padding=padding),
        #     nn.Conv2d(in_channels=hidden_dim/2, 
        #     out_channels=hidden_dim/4,
        #     kernel_size=kernel_size,
        #     padding=padding),
        #     nn.Conv2d(in_channels=hidden_dim/4, 
        #     out_channels=input_dim,
        #     kernel_size=kernel_size,
        #     padding=padding)
        # )


        # self.conv = nn.Sequential()
        # self.conv.add_module(
        #     "conv1", nn.Conv2d(
        #         in_channels=hidden_dim, out_channels=hidden_dim / 2,
        #         kernel_size=kernel_size,
        #         padding=padding)
        # )

        # self.conv.add_module(
        #     "conv2", nn.Conv2d(
        #         in_channels=hidden_dim / 2, out_channels=hidden_dim / 4,
        #         kernel_size=kernel_size,
        #         padding=padding)
        # )

        # self.conv.add_module(
        #     "conv3", nn.Conv2d(
        #         in_channels=hidden_dim / 4, out_channels=input_dim,
        #         kernel_size=kernel_size,
        #         padding=padding)
        # )
    def print_device(self):
        print(f"self.device: {self.device}")

    def forward(self, x):
        #print(f"encdec device: {self.device}")

        conv_lstm_output = self.mod(x)
        # Last hidden state will be used for predicting next frame
        pred_frame = self.conv1(conv_lstm_output[:, :, -1, :, :])
        return pred_frame

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        return {"optimizer": optimizer, 
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss"
                }

    def training_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        tgt_frames = tgt_frames[:, :, 0, :, :]
        pred_frame = self.forward(ctx_frames)
        loss = self.mse(pred_frame, tgt_frames)
        self.log('train_loss', loss,
                 prog_bar=False)
        return loss

        
    def validation_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        tgt_frame = tgt_frames[:, :, 0, :, :]
        # loss = self.mse(pred_frame, tgt_frames)
        # self.log('train_loss', loss,
        #          prog_bar=False)
        # return loss
        pred_frame = self.forward(ctx_frames)
        loss = self.mse(tgt_frame, pred_frame)
        
        tgt_frame = tgt_frame.unsqueeze(2)
        pred_frame = pred_frame.unsqueeze(2)

        ssim = self.ssim(tgt_frame, pred_frame)
        psnr = self.psnr(tgt_frame, pred_frame)
        self.log_dict(
            {"val_loss": loss,
             "val_ssim": ssim,
             "val_psnr": psnr
            }, on_step=False, on_epoch=True, prog_bar=False) 

        return ctx_frames, tgt_frame, pred_frame

    def validation_epoch_end(self, validation_step_outputs):
        # Add plot to logger every 5 epochs
        if (self.current_epoch+1) % 5 == 0:
            # first batch in validation dataset
            batch_ctx, batch_tgt, batch_pred = validation_step_outputs[0]
            # first video
            ctx_frames = batch_ctx[0]    # C x F x H x W
            tgt_frame = batch_tgt[0]     # C x H x W
            #tgt_frame = tgt_frame.unsqueeze(1)
            tgt_frames = tgt_frame.expand(-1, 5, -1, -1)
            
            pred_frame = batch_pred[0]  # C x H x W
           # pred_frame = pred_frame.unsqueeze(1)
            pred_frames = pred_frame.expand(-1, 5, -1, -1)  # C x F x H x W
    
            img = make_plot_image(ctx_frames, tgt_frames,
                                  pred_frames, epoch=self.current_epoch+1)
            
            tb = self.logger.experiment
            tb.add_image("val_predictions", img, global_step=self.current_epoch)


