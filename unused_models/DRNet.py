import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# from models.drnet_classes import *
# from models.metrics import *
# from models.logging_utils import *
from models import *

class DRNetMain(pl.LightningModule):
    def __init__(self, channels=3, content_dim=128, pose_dim=5,
                 discriminator_dim=100,
                 learning_rate=1e-3,
                 alpha=1,
                 beta=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta

        self.content_encoder = Encoder(channels, content_dim)
        self.pose_encoder = Encoder(channels, pose_dim)
        self.decoder = Decoder(content_dim + pose_dim, channels, use_skip=True)
        
        self.scene_discriminator = SceneDiscriminator(pose_dim, discriminator_dim)
        # self.lstm_pose_generator = LSTMPoseGenerator(content_dim + pose_dim, lstm_dim,
        #                               pose_dim, batch_size)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def configure_optimizers(self):
        # opt_content_enc = torch.optim.Adam(self.content_encoder.parameters(), 
        #                                    lr=self.learning_rate)
        
        # opt_pose_enc = torch.optim.Adam(self.pose_encoder.parameters(), 
        #                                 lr=self.learning_rate)
        # opt_decoder = torch.optim.Adam(self.decoder.parameters(), 
        #                                lr=self.learning_rate)


        opt_scene_dis = torch.optim.Adam(self.scene_discriminator.parameters(), 
                                         lr=self.learning_rate)

        main_network_params = list(self.content_encoder.parameters()) +\
                              list(self.pose_encoder.parameters()) +\
                              list(self.decoder.parameters())

        opt_main_network = torch.optim.Adam(main_network_params, 
                                            lr=self.learning_rate)


        
        return [opt_scene_dis, opt_main_network], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        ctx_frames, tgt_frames = batch  # BS x C x F x H x W
        batch_size = ctx_frames.size(0)
        combined_frames = torch.cat([ctx_frames, tgt_frames], axis=2)

        ########################################
        ###### Optimize scene discriminator ####
        ########################################
        if optimizer_idx == 0:
            first_frames = combined_frames[:, :, 0]
            second_frames = combined_frames[:, :, 1]
            first_frames_poses = self.pose_encoder(first_frames)[0].detach()
            second_frames_poses = self.pose_encoder(second_frames)[0].detach()

            half_batch_size = batch_size // 2
            reorder = torch.randperm(half_batch_size).cuda()
            second_frames_poses[:half_batch_size] = second_frames_poses[reorder]

            target = torch.cuda.FloatTensor(batch_size, 1)
            target[:half_batch_size] = 1
            target[half_batch_size:] = 0   

            out = self.scene_discriminator(first_frames_poses, second_frames_poses)
            bce = self.bce(out, Variable(target))
            acc = (out[:half_batch_size].gt(0.5).sum() + out[half_batch_size:].le(0.5).sum()) / batch_size

            self.log_dict(
                {"train_discriminator_bce": bce,
                 "train_discriminator_acc": acc, 
                }, on_step=False, on_epoch=True, prog_bar=False)
            
            return bce

        ########################################
        ###### Optimize main network  ##########
        ########################################
        if optimizer_idx == 1:
            first_frames = combined_frames[:, :, 0]
            second_frames = combined_frames[:, :, 1]
            third_frames = combined_frames[:, :, 2]
            fourth_frames = combined_frames[:, :, 3]

            first_frames_con, skip1 = self.content_encoder(first_frames)
            second_frames_con, skip2 = self.content_encoder(second_frames)
            second_frames_con = second_frames_con.detach()

            third_frames_pose = self.pose_encoder(third_frames)[0]
            forth_frames_pose = self.pose_encoder(fourth_frames)[0].detach()

            # similarity loss
            sim_loss = self.mse(first_frames_con, second_frames_con)

            # recon loss
            rec = self.decoder(first_frames_con, skip1, third_frames_pose)
            rec_loss = self.mse(rec, third_frames)

            # scene discriminator loss
            target = torch.cuda.FloatTensor(ctx_frames.size(0), 1).fill_(0.5)
            out = self.scene_discriminator(third_frames_pose, forth_frames_pose)
            sd_loss = self.bce(out, Variable(target))
            
            # full loss
            loss = sim_loss + self.alpha*rec_loss + self.beta*sd_loss

            self.log_dict({
                "train_full_loss": loss,
                "train_sim_loss": sim_loss,
                "train_rec_loss": rec_loss,
                "train_sd_loss": sd_loss
            }, on_step=False, on_epoch=True, prog_bar=False)

            return loss

class DRNetLSTM(pl.LightningModule):
    def __init__(self, content_encoder, pose_encoder,
                 content_dim=128, pose_dim=5,
                 lstm_hidden_units=256,
                 layers=1,
                 batch_size=16,
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['content_encoder', 'pose_encoder'])
        # self.automatic_optimization = False

        self.mse = nn.MSELoss()
        self.learning_rate = learning_rate
        self.training_loss = None

        self.content_encoder = content_encoder
        self.pose_encoder = pose_encoder
        self.lstm_pose_generator = LSTMPoseGenerator(content_dim + pose_dim,           
                                                     lstm_hidden_units, 
                                                     pose_dim,
                                                     batch_size,
                                                     layers)

        self.lstm_pose_generator.init_hidden()

    def configure_optimizers(self):
        opt_lstm_pose_generator = torch.optim.Adam(self.lstm_pose_generator.parameters(), 
                                                   lr=self.learning_rate)

        return opt_lstm_pose_generator

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward(retain_graph=True)

    # def optimizer_step(self, epoch, batch_idx, 
    #                    optimizer, optimizer_idx,
    #                    optimizer_closure, **kwarg):

    #     optimizer.step()
    #     if self.training_loss is not None:
    #         optimizer.zero_grad()
    #         self.training_loss.backward()
    #         optimizer.step(closure=optimizer_closure)
    #         self.training_loss = None
    #     else:
    #         optimizer_closure()

    def training_step(self, batch, batch_idx):

        # opt = self.optimizers()
        # opt.zero_grad()
        self.lstm_pose_generator.zero_grad()
        self.lstm_pose_generator.init_hidden()

        ctx_frames, tgt_frames = batch  # BS x C x F x H x W
        num_ctx_frames = ctx_frames.size(2)
        num_tgt_frames = tgt_frames.size(2)

        # compute the fixed content feature from the last frame
        last_frames_content = self.content_encoder(ctx_frames[:, :, -1])[0].detach()

        # compute the pose features for each of the time step
        all_frames_poses = [self.pose_encoder(ctx_frames[:, :, i])[0].detach() for i in range(num_ctx_frames)] + [self.pose_encoder(tgt_frames[:, :, i])[0].detach() for i in range(num_tgt_frames)]

        mse = 0.

        for i in range(1, num_ctx_frames+num_tgt_frames):
            pred = self.lstm_pose_generator(torch.cat([last_frames_content, all_frames_poses[i-1]], dim=1))
            # mse += F.mse_loss(pred, all_frames_poses[i].squeeze())
            mse += self.mse(pred, all_frames_poses[i].squeeze())

        training_loss = torch.autograd.Variable(mse).requires_grad_(True)
        # self.training_loss = torch.autograd.Variable(mse).requires_grad_(True)
        # self.training_loss = mse.clone()
        # self.training_loss = mse

        # self.manual_backward(mse)
        # opt.step()

        self.log("train_lstm_loss", mse, on_step=False, 
                 on_epoch=True, prog_bar=True)

        return training_loss
