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

input_dim=3
hidden_dim=64
output_dim=3 
kernel_size=(3, 3)
bias=True
num_tgt_frames=5
learning_rate=1e-3

model = EncoderDecoderConvLSTM(input_dim, hidden_dim, output_dim, kernel_size, bias,
                               num_tgt_frames,
                               learning_rate=1e-3)

moving_mnist = TwoColourMovingMNISTDataModule(batch_size, num_ctx_frames, num_tgt_frames,
                                              split_ratio=[0.2, 0.05, 0.75])

logger = TensorBoardLogger('./logs', 'ConvLSTM_RGB')

trainer = pl.Trainer(gpus=4, 
                     strategy=DDPStrategy(find_unused_parameters=False),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, moving_mnist)