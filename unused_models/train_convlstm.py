import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *

# Configs
batch_size = 16
learning_rate = 1e-3
epochs = 50

num_ctx_frames = 5
num_tgt_frames = 5

input_dim = 1
hidden_dim = 64
frame_dim = (64, 64)
kernel_size = (3, 3)
padding = 1
num_lstm_layers = 3

model = EncoderDecoderConvLSTM(input_dim, hidden_dim, frame_dim,
                               kernel_size, padding,
                               num_lstm_layers, learning_rate=1e-4).cuda()

print(model.device)
moving_mnist = MovingMNISTDataModule(batch_size, num_ctx_frames, num_tgt_frames,
                                     split_ratio=[0.7, 0.15, 0.15])

logger = TensorBoardLogger('./logs', 'ConvLSTM')

trainer = pl.Trainer(gpus=4, 
                     strategy=DDPStrategy(find_unused_parameters=False),
                     num_sanity_val_steps=0,
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, moving_mnist)