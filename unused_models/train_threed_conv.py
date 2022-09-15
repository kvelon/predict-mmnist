import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *

# Configs
batch_size = 16
learning_rate = 1e-3
epochs = 100

num_ctx_frames = 5
num_tgt_frames = 5

model = ThreeDConv(learning_rate)
moving_mnist = MovingMNISTDataModule(batch_size, num_ctx_frames, num_tgt_frames)

logger = TensorBoardLogger('./logs', 'ThreeDConv')

trainer = pl.Trainer(gpus=4, 
                     strategy=DDPStrategy(find_unused_parameters=False),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, moving_mnist)