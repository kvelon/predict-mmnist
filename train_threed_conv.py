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

model = ThreeDConv_1to9(learning_rate)
moving_mnist = TwoColourMovingMNISTDataModule(batch_size, 
                                              num_ctx_frames, num_tgt_frames,
                                              split_ratio=split_ratio)

logger = TensorBoardLogger('./logs', 'ThreeDConv_RGB')

trainer = pl.Trainer(gpus=1, 
                    #  strategy=DDPStrategy(find_unused_parameters=False),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, moving_mnist)