import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.metrics import *
from models.logging_utils import *

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, stride):
        super().__init__()

        self.num_hidden = num_hidden
        self.padding = kernel_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=False),
        )

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        # print(f"f_t: {f_t.shape}")
        # print(f"c_t: {c_t.shape}")
        # print(f"i_t: {i_t.shape}")
        # print(f"g_t: {g_t.shape}")
        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

class PredRNN(pl.LightningModule):
    def __init__(self, input_channels,
                 num_hidden,
                 num_ctx_frames,
                 num_tgt_frames,
                 kernel_size,
                 stride,
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.num_hidden = num_hidden
        self.num_layers = len(num_hidden)
        self.num_ctx_frames = num_ctx_frames
        self.num_tgt_frames = num_tgt_frames
        self.learning_rate = learning_rate 

        # self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()
        self.ssim = SSIM()
        self.psnr = PSNR()

        cell_list = []


        for i in range(self.num_layers):
            in_channel = input_channels if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], kernel_size, stride)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[self.num_layers - 1], 
                                   input_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        B, C, F, H, W = x.shape

        next_frames = []
        h_t = []
        c_t = []

        # Initialize hidden states and cell states
        for i in range(self.num_layers):
            zeros = torch.zeros([B, self.num_hidden[i], H, W],
                                device=self.device)
            h_t.append(zeros)
            c_t.append(zeros)

        # Initialize memory state
        memory = torch.zeros([B, self.num_hidden[0], H, W],
                             device=self.device)

        for t in range(F - 1):
            frame = x[:, :, t]
            h_t[0], c_t[0], memory = self.cell_list[0](frame, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=2)
        return next_frames

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        return {"optimizer": optimizer, 
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss"
                }

    def training_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        all_frames = torch.cat([ctx_frames, tgt_frames], dim=2)
        next_frames = self.forward(all_frames)
        loss = self.loss(next_frames, all_frames[:, :, 1:])
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ctx_frames, tgt_frames = batch
        all_frames = torch.cat([ctx_frames, tgt_frames], dim=2)

        next_frames = self.forward(all_frames)

        loss = self.loss(next_frames, all_frames[:, :, 1:])
        ssim = self.ssim(next_frames, all_frames[:, :, 1:])
        psnr = self.psnr(next_frames, all_frames[:, :, 1:])
        self.log_dict(
            {"val_loss": loss,
             "val_ssim": ssim,
             "val_psnr": psnr
            }, on_step=False, on_epoch=True, prog_bar=False)  
        
        pred_frames = next_frames[:, :, self.num_ctx_frames-1:]

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