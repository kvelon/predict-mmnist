import torch
import torch.nn as nn

class Conv3dBlock(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)):
        super().__init__()

        self.mod = nn.Sequential(
            nn.Conv3d(input_channels, output_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.mod(x)

class Conv3dDownsample(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        output_channels = input_channels * 2

        self.mod = nn.Sequential(
            nn.Conv3d(input_channels, output_channels,
                      stride=(1, 2, 2),
                      kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.mod(x)

class ConvTranspose3dBlock(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)):
        super().__init__()

        self.mod = nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels,
                               kernel_size=kernel_size, padding=padding, 
                               stride=stride),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.mod(x)

class ConvTranspose3dUpsample(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        output_channels = input_channels // 2

        self.mod = nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels,
                               stride=(1,2,2),kernel_size=(3,3,3), padding=(1,1,1), output_padding=(0,1,1)),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.mod(x)

class ThreeDConvDeepTwo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mod = nn.Sequential(
            Conv3dBlock(1, 8), 
            Conv3dBlock(8, 8), Conv3dBlock(8, 8), 
            Conv3dDownsample(8),
            Conv3dBlock(16, 16), Conv3dBlock(16,16),
            Conv3dDownsample(16),
            Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            ConvTranspose3dUpsample(32),
            ConvTranspose3dBlock(16, 16), ConvTranspose3dBlock(16, 16),
            ConvTranspose3dUpsample(16),
            ConvTranspose3dBlock(8, 8), ConvTranspose3dBlock(8, 8),
            nn.Conv3d(8, 1, kernel_size = (1,1,1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mod(x)

class ThreeDConvWideFourDeepThree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mod = nn.Sequential(
            Conv3dBlock(3, 32), 
            Conv3dBlock(32, 32), Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            Conv3dDownsample(32),
            Conv3dBlock(64, 64), Conv3dBlock(64, 64), Conv3dBlock(64, 64),
            Conv3dDownsample(64),
            Conv3dBlock(128, 128), Conv3dBlock(128, 128),
            ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128),
            ConvTranspose3dUpsample(128),
            ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64),
            ConvTranspose3dUpsample(64),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            nn.Conv3d(32, 3, kernel_size = (1,1,1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mod(x)

class ThreeDConvWideFourDeepThreeAutoreg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mod = nn.Sequential(
            Conv3dBlock(1, 32), 
            Conv3dBlock(32, 32), Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            Conv3dDownsample(32),
            Conv3dBlock(64, 64), Conv3dBlock(64, 64), Conv3dBlock(64, 64),
            Conv3dDownsample(64),
            Conv3dBlock(128, 128), Conv3dBlock(128, 128),
            ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128),
            ConvTranspose3dUpsample(128),
            ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64),
            ConvTranspose3dUpsample(64),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            nn.Conv3d(32, 1, kernel_size = (5,1,1)),
            nn.Sigmoid(),
        )

    def pred_frame(self, x):
        return self.mod(x)

    def forward(self, x):
        ctx_frames = x.clone()
        num_ctx_frames = ctx_frames.shape[2]
        
        for t in range(num_ctx_frames):
            next_frame = self.pred_frame(ctx_frames[:, :, t:t+num_ctx_frames, :, :])
            ctx_frames = torch.cat([ctx_frames, next_frame], dim=2)
        return ctx_frames[:, :, num_ctx_frames:, :, :]

class ThreeDConvWideFourDeepThreeSkip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mod1 = nn.Sequential(
            Conv3dBlock(1, 32), 
            Conv3dBlock(32, 32), Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            Conv3dDownsample(32),
            Conv3dBlock(64, 64), Conv3dBlock(64, 64), Conv3dBlock(64, 64)
        )
        
        self.mod2 = nn.Sequential(
            Conv3dDownsample(64),
            Conv3dBlock(128, 128), Conv3dBlock(128, 128), Conv3dBlock(128, 128),
            ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128),
            ConvTranspose3dUpsample(128)
        )

        self.mod1_2 = nn.Sequential(
            ConvTranspose3dBlock(64 * 2, 64)
        )
         
        self.mod3 = nn.Sequential(
            ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64),
            ConvTranspose3dUpsample(64),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            nn.Conv3d(32, 1, kernel_size = (1,1,1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        after_mod1 = self.mod1(x)
        after_mod2 = self.mod2(after_mod1)
        after_mod1_2 = self.mod1_2(torch.cat([after_mod1, after_mod2], dim=1))
        output = self.mod3(after_mod1_2)
        return output

class ThreeDConvWideFourDeepThreeSkipAutoreg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mod1 = nn.Sequential(
            Conv3dBlock(1, 32), 
            Conv3dBlock(32, 32), Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            Conv3dDownsample(32),
            Conv3dBlock(64, 64), Conv3dBlock(64, 64), Conv3dBlock(64, 64)
        )
        
        self.mod2 = nn.Sequential(
            Conv3dDownsample(64),
            Conv3dBlock(128, 128), Conv3dBlock(128, 128), Conv3dBlock(128, 128),
            ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128),
            ConvTranspose3dUpsample(128)
        )

        self.mod1_2 = nn.Sequential(
            ConvTranspose3dBlock(64 * 2, 64)
        )
         
        self.mod3 = nn.Sequential(
            ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64),
            ConvTranspose3dUpsample(64),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            nn.Conv3d(32, 1, kernel_size = (5,1,1)),  # from 5 frames predicts 1 frame
            nn.Sigmoid(),
        )

    def pred_frame(self, x):
        after_mod1 = self.mod1(x)
        after_mod2 = self.mod2(after_mod1)
        after_mod1_2 = self.mod1_2(torch.cat([after_mod1, after_mod2], dim=1))
        output = self.mod3(after_mod1_2)
        return output

    def forward(self, x):
        ctx_frames = x.clone()
        num_ctx_frames = ctx_frames.shape[2]

        for t in range(num_ctx_frames):
            next_frame = self.pred_frame(ctx_frames[:, :, t:t+num_ctx_frames, :, :])
            ctx_frames = torch.cat([ctx_frames, next_frame], dim=2)
        return ctx_frames[:, :, num_ctx_frames:, :, :]

class ThreeDConvWideFourDeepThreeSkipAutoregRGB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mod1 = nn.Sequential(
            Conv3dBlock(3, 32), 
            Conv3dBlock(32, 32), Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            Conv3dDownsample(32),
            Conv3dBlock(64, 64), Conv3dBlock(64, 64), Conv3dBlock(64, 64)
        )
        
        self.mod2 = nn.Sequential(
            Conv3dDownsample(64),
            Conv3dBlock(128, 128), Conv3dBlock(128, 128), Conv3dBlock(128, 128),
            ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128),
            ConvTranspose3dUpsample(128)
        )

        self.mod1_2 = nn.Sequential(
            ConvTranspose3dBlock(64 * 2, 64)
        )
         
        self.mod3 = nn.Sequential(
            ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64),
            ConvTranspose3dUpsample(64),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            nn.Conv3d(32, 3, kernel_size = (5,1,1)),  # from 5 frames predicts 1 frame
            nn.Sigmoid(),
        )

    def pred_frame(self, x):
        after_mod1 = self.mod1(x)
        after_mod2 = self.mod2(after_mod1)
        after_mod1_2 = self.mod1_2(torch.cat([after_mod1, after_mod2], dim=1))
        output = self.mod3(after_mod1_2)
        return output

    def forward(self, x):
        ctx_frames = x.clone()
        num_ctx_frames = ctx_frames.shape[2]

        for t in range(num_ctx_frames):
            next_frame = self.pred_frame(ctx_frames[:, :, t:t+num_ctx_frames, :, :])
            ctx_frames = torch.cat([ctx_frames, next_frame], dim=2)
        return ctx_frames[:, :, num_ctx_frames:, :, :]

class ThreeDConvWideFourDeepThreeSkipRGB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mod1 = nn.Sequential(
            Conv3dBlock(3, 32), 
            Conv3dBlock(32, 32), Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            Conv3dDownsample(32),
            Conv3dBlock(64, 64), Conv3dBlock(64, 64), Conv3dBlock(64, 64)
        )
        
        self.mod2 = nn.Sequential(
            Conv3dDownsample(64),
            Conv3dBlock(128, 128), Conv3dBlock(128, 128), Conv3dBlock(128, 128),
            ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128), ConvTranspose3dBlock(128, 128),
            ConvTranspose3dUpsample(128)
        )

        self.mod1_2 = nn.Sequential(
            ConvTranspose3dBlock(64 * 2, 64)
        )
         
        self.mod3 = nn.Sequential(
            ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64), ConvTranspose3dBlock(64, 64),
            ConvTranspose3dUpsample(64),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            nn.Conv3d(32, 3, kernel_size = (1,1,1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        after_mod1 = self.mod1(x)
        after_mod2 = self.mod2(after_mod1)
        after_mod1_2 = self.mod1_2(torch.cat([after_mod1, after_mod2], dim=1))
        output = self.mod3(after_mod1_2)
        return output
