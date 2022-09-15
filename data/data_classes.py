import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets


class MovingMNISTDataset(Dataset):
    def __init__(self, num_x_frames, num_y_frames, data_path='/home/kelvinfung/Documents/bounce-digits/data/mnist_test_seq.npy'):
        # F: Frames, L: Length of Data
        arr = np.load(data_path).reshape(-1, 64, 64)  # (F x L) x H x W
        arr = np.transpose(arr, (1, 2, 0))  # H x W x (FxL)

        arr = transforms.ToTensor()(arr).reshape(20, -1, 64, 64)  # F x L x H x W
        self.arr = torch.unsqueeze(arr, dim=0)  # C x F x L x H x W
        
        #self.arr = self.zero_centre(arr)
        self.x_frames = self.arr[:, :num_x_frames, :, :, :]
        self.y_frames = self.arr[:, num_x_frames:num_x_frames+num_y_frames, :, :, :]

    def zero_centre(self, arr):
        return arr - arr.mean()

    def __len__(self):
        return self.x_frames.shape[2]
    
    def __getitem__(self, idx):
        return self.x_frames[:, :, idx, :, :], self.y_frames[:, :, idx, :, :]

class MovingMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_x_frames, num_y_frames, data_path='/home/kelvinfung/Documents/bounce-digits/data/mnist_test_seq.npy', split_ratio=[0.7, 0.15, 0.15]):
        super().__init__()
        self.batch_size = batch_size
        self.num_x_frames = num_x_frames
        self.num_y_frames = num_y_frames
        self.data_path = data_path
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        full_dataset = MovingMNISTDataset(self.num_x_frames,
                                          self.num_y_frames,
                                          self.data_path)
        split = [int(len(full_dataset) * r) for r in self.split_ratio]
        train, val, test = random_split(full_dataset, split,
                                        generator=torch.Generator().manual_seed(42))
        self.train = train
        self.val = val
        self.test = test

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.batch_size,
                          shuffle = True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val,
                         batch_size = self.batch_size,
                         shuffle = False,
                         num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size = self.batch_size,
                          shuffle = False,
                          num_workers=8)

##########################################
### Dataset classes for RGB MNIST ########
##########################################

class RGBMovingMNISTDataset(Dataset):
    def __init__(self, num_context_frames, num_target_frames, data_path='data/mnist_rgb.npy'):
        # F: Frames, L: Length of Data
        arr = np.load(data_path)  # L x F x H x W x C
        self.arr = arr
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.context_frames = self.arr[:, :num_context_frames, :, :, :]
        self.target_frames = self.arr[:, num_context_frames:num_context_frames+num_target_frames, :, :, :]

        self.transform = transforms.ToTensor()

    def zero_centre(self, arr):
        return arr - arr.mean()

    def __len__(self):
        return self.target_frames.shape[0]
    
    def __getitem__(self, idx):
        f, h, w, c = self.context_frames[0].shape
        context_frames = torch.zeros(3, self.num_context_frames, h, w)
        target_frames = torch.zeros(3, self.num_target_frames, h, w)

        for i in range(self.num_context_frames):
            frame = self.context_frames[idx, i, :, :, :]  # H x W x C
            ts = self.transform(frame)  # C x H x W
            context_frames[:, i, :, :] = ts

        for j in range(self.num_target_frames):
            frame = self.target_frames[idx, j, :, :, :]  # H x W x C
            ts = self.transform(frame)  # C x H x W
            target_frames[:, j, :, :] = ts
        
        return context_frames, target_frames

class RGBMovingMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_context_frames, num_target_frames, data_path='./data/mnist_rgb.npy', split_ratio=[0.7, 0.15, 0.15]):
        super().__init__()
        self.batch_size = batch_size
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.data_path = data_path
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        full_dataset = RGBMovingMNISTDataset(self.num_context_frames,
                                             self.num_target_frames,
                                             self.data_path)
        split = [int(len(full_dataset) * r) for r in self.split_ratio]
        train, val, test = random_split(full_dataset, split,
                                        generator=torch.Generator().manual_seed(42))
        self.train = train
        self.val = val
        self.test = test

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.batch_size,
                          shuffle = True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val,
                         batch_size = self.batch_size,
                         shuffle = False,
                         num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size = self.batch_size,
                          shuffle = False,
                          num_workers=8)

    
################################################
### Dataset classes for two-coloured MNIST #####
################################################

class TwoColourMovingMNISTDataset(Dataset):
    def __init__(self, num_ctx_frames, num_tgt_frames, data_path='/home/kelvinfung/Documents/bounce-digits/data/'):

        self.num_ctx_frames = num_ctx_frames
        self.num_tgt_frames = num_tgt_frames
        self.seq_len = num_ctx_frames + num_tgt_frames
        self.num_digits = 2  
        self.image_size = 128
        self.step_length = 0.1
        self.digit_size = 64
        self.data = datasets.MNIST(
            data_path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()])
        )
        self.N = len(self.data)

    def set_seed(self, seed=42):
        np.random.seed(seed)
    
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)  # Data's index used to set seed
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      3),
                     dtype=np.float32)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-8, 8)
            dy = np.random.randint(-8, 8)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    dy = -dy
                elif sy >= image_size-digit_size:
                    sy = image_size-digit_size-1
                    dy = -dy
                    
                if sx < 0:
                    sx = 0 
                    dx = -dx
                elif sx >= image_size-digit_size:
                    sx = image_size-digit_size-1
                    dx = -dx
                   
                x[t, sy:sy+digit_size, sx:sx+digit_size, n] = np.copy(digit.numpy())
                sy += dy
                sx += dx

        # pick one digit to be in front
        front = np.random.randint(self.num_digits)
        for cc in range(self.num_digits):
            if cc != front:
                x[:, :, :, cc][x[:, :, :, front] > 0] = 0

        # transform data to tensor
        context_frames = torch.zeros(3, self.num_ctx_frames, image_size, image_size)
        target_frames = torch.zeros(3, self.num_tgt_frames, image_size, image_size)

        for i in range(self.num_ctx_frames):
            ts = transforms.ToTensor()(x[i])
            context_frames[:, i] = ts

        for j in range(self.num_tgt_frames):
            ts = transforms.ToTensor()(x[self.num_ctx_frames + j])
            target_frames[:, j] = ts
        
        # C x F x H x W
        return context_frames, target_frames

class TwoColourMovingMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_ctx_frames, num_tgt_frames, 
                 data_path='/home/kelvinfung/Documents/bounce-digits/data/',
                 split_ratio=[0.7, 0.15, 0.15]):
        super().__init__()
        self.batch_size = batch_size
        self.num_ctx_frames = num_ctx_frames
        self.num_tgt_frames = num_tgt_frames        
        self.data_path = data_path
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        full_dataset = TwoColourMovingMNISTDataset(self.num_ctx_frames,
                                                   self.num_tgt_frames,
                                                   self.data_path)
        split = [int(len(full_dataset) * r) for r in self.split_ratio]
        self.split = split
        train, val, test = random_split(full_dataset, split,
                                        generator=torch.Generator().manual_seed(42))
        self.train = train
        self.val = val
        self.test = test

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.batch_size,
                          shuffle = True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val,
                         batch_size = self.batch_size,
                         shuffle = False,
                         num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size = self.batch_size,
                          shuffle = False,
                          num_workers=4)
