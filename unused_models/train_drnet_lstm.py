import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *
from unused_models.DRNet import *

# Configs
ckpt_path = "./logs/DRNetMain_RGB/version_0/checkpoints/epoch=49-step=12500.ckpt"

batch_size = 16
learning_rate = 1e-4
epochs = 50

num_ctx_frames = 5
num_tgt_frames = 5

channels=3
content_dim=128
pose_dim=5
                 
discriminator_dim=100
lstm_hidden_units=256
learning_rate=1e-3
alpha=1
beta=0.1
layers=1

# Prepare trained model
main_model = DRNetMain(channels=channels, 
                       content_dim=content_dim, 
                       pose_dim=pose_dim,
                       discriminator_dim=discriminator_dim,
                       learning_rate=learning_rate,
                       alpha=alpha,
                       beta=beta)

main_model = main_model.load_from_checkpoint(ckpt_path)
content_encoder = main_model.content_encoder
pose_encoder = main_model.pose_encoder

model = DRNetLSTM(content_encoder, pose_encoder,
                  content_dim=content_dim, 
                  pose_dim=pose_dim,
                  lstm_hidden_units=lstm_hidden_units,
                  layers=1,
                  batch_size=batch_size)

moving_mnist = TwoColourMovingMNISTDataModule(batch_size, 
                                              num_ctx_frames, 
                                              num_tgt_frames,
                                              split_ratio=[0.4, 0.1, 0.5])

logger = TensorBoardLogger('./logs', 'DRNetLSTM_RGB')

trainer = pl.Trainer(gpus=1, 
                     strategy=DDPStrategy(find_unused_parameters=True),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger,
                     detect_anomaly=True)

trainer.fit(model, moving_mnist)