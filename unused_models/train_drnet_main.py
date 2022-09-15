import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from DRNet import *
from data.data_classes import *

# Configs
batch_size = 16
learning_rate = 1e-4
epochs = 50

num_ctx_frames = 5
num_tgt_frames = 5

channels=3
content_dim=128
pose_dim=5
                 
discriminator_dim=100
learning_rate=1e-3
alpha=1
beta=0.1

# model = DRNetMain(channels=3, content_dim=128, pose_dim=5,
#                  discriminator_dim=100,
#                  learning_rate=1e-3,
#                  alpha=1,
#                  beta=0.1)

model = DRNetMain(channels=channels, 
                  content_dim=content_dim, 
                  pose_dim=pose_dim,
                  discriminator_dim=discriminator_dim,
                  learning_rate=learning_rate,
                  alpha=alpha,
                  beta=beta)

moving_mnist = TwoColourMovingMNISTDataModule(batch_size, 
                                              num_ctx_frames, 
                                              num_tgt_frames,
                                              split_ratio=[0.4, 0.1, 0.5])

logger = TensorBoardLogger('./logs', 'DRNetMain_RGB')

trainer = pl.Trainer(gpus=2, 
                     strategy=DDPStrategy(find_unused_parameters=True),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, moving_mnist)