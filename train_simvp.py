import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *

# Configs
batch_size = 4
learning_rate = 1e-3
epochs = 100

num_ctx_frames = 1
num_tgt_frames = 9
split_ratio=[0.4, 0.1, 0.5]

hid_s=64
hid_t=256
N_s=4
N_t=8
kernel_sizes=[3,5,7,11]
groups=4

channels = 3
height = 128
width = 128
input_shape = (channels, num_ctx_frames, height, width)

model = SimVP_1to9(input_shape=input_shape, 
                   hid_s=hid_s, hid_t=hid_t, 
                   N_s=N_s, N_t=N_t,
                   kernel_sizes=kernel_sizes, 
                   groups=groups,
                   learning_rate=learning_rate)

moving_mnist = TwoColourMovingMNISTDataModule(batch_size, 
                                              num_ctx_frames, num_tgt_frames,
                                              split_ratio=split_ratio)

logger = TensorBoardLogger('./logs', 'SimVP')

trainer = pl.Trainer(gpus=1, 
                    #  strategy=DDPStrategy(find_unused_parameters=False),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, moving_mnist)