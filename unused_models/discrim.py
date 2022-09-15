import torch
import torch.nn as nn

from unused_models.futuregan_classes import *

config = {
    'nframes_pred': 5,
    'nframes_in' : 5,
    'batch_norm' : False,
    'w_norm' : True,
    'loss' : 'wgan_gp',
    'd_gdrop' : False,
    'padding' : 'zero',
    'lrelu' : True,
    'd_sigmoid' : False,
    'nz' : 512,            # dim of input noise vector z    
    'nc' : 1,              # number of channels
    'ndf' : 512,           # discriminator first layer's feature dim
    'd_cond' : True
}

dis = Discriminator(config)

N, C, F, H, W = 50, 1, 5, 128, 128
sample_batch = torch.randn(N, C, F, H, W)
results = dis(sample_batch)
print("Discriminator results complete.")
print(f"results shape: {results.shape}")