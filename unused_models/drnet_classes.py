from turtle import forward
from cv2 import _InputArray_FIXED_SIZE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BasicConv2d(nn.Module):
    # B, C_in, H, W -> B, C_out, H', W'
    def __init__(self, in_channels, out_channels,
                  kernel_size, stride,
                  padding, transpose=False):
        super().__init__()
        if not transpose:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels, 
                                  kernel_size=kernel_size,
                                  stride=stride, 
                                  padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, 
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding
            )
        self.norm = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.nonlinearity(y)
        return y

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=False):
        super().__init__()
        
        self.normalize = normalize
        self.block1 = BasicConv2d(in_channels, 64,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1)
        self.block2 = BasicConv2d(64, 128,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1)

        self.block3 = BasicConv2d(128, 256,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1)

        self.block4 = BasicConv2d(256, 512,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1)

        self.block5 = BasicConv2d(512, 512,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1)
        
        self.block6 = nn.Sequential(nn.Conv2d(512, out_channels,
                                              kernel_size=4,
                                              stride=1,
                                              padding=0),
                                    nn.BatchNorm2d(out_channels),
                                    nn.Tanh()
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        out6 = self.block6(out5)
        
        if self.normalize:
            out6 = F.normalize(out6, p=2)

        return out6, [out1, out2, out3, out4, out5]    

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_skip=False):
        super().__init__()
        
        self.use_skip = use_skip

        mul_factor = 2 if use_skip else 1
        self.mul_factor = mul_factor

        self.block1 = BasicConv2d(in_channels, 512,
                                  kernel_size=4, 
                                  stride=1,
                                  padding=0,
                                  transpose=True)

        self.block2 = BasicConv2d(512*mul_factor, 512,
                                  kernel_size=4, 
                                  stride=2,
                                  padding=1,
                                  transpose=True)

        self.block3 = BasicConv2d(512*mul_factor, 256,
                                  kernel_size=4, 
                                  stride=2,
                                  padding=1,
                                  transpose=True)

        self.block4 = BasicConv2d(256*mul_factor, 128,
                                  kernel_size=4, 
                                  stride=2,
                                  padding=1,
                                  transpose=True)

        self.block5 = BasicConv2d(128*mul_factor, 64,
                                  kernel_size=4, 
                                  stride=2,
                                  padding=1,
                                  transpose=True)             

        self.block6 = nn.Sequential(
            nn.ConvTranspose2d(64*mul_factor, out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Sigmoid()
        )                  

    def forward(self, content, skip, pose):

        inp1 = torch.cat([content, pose], dim=1)
        out1 = self.block1(inp1)

        if self.use_skip:
            inp2 = torch.cat([out1, skip[4]], dim=1)
            out2 = self.block2(inp2)

            inp3 = torch.cat([out2, skip[3]], dim=1)
            out3 = self.block3(inp3)

            inp4 = torch.cat([out3, skip[2]], dim=1)
            out4 = self.block4(inp4)

            inp5 = torch.cat([out4, skip[1]], dim=1)
            out5 = self.block5(inp5)

            inp6 = torch.cat([out5, skip[0]], dim=1)
            out6 = self.block6(inp6)
        
        else: 
            out2 = self.block2(out1)
            out3 = self.block2(out2)
            out4 = self.block2(out3)
            out5 = self.block2(out4)
            out6 = self.block2(out5)

        return out6

class SceneDiscriminator(nn.Module):
    def __init__(self, pose_channels, hidden_units=100) -> None:
        super().__init__()

        self.pose_channels = pose_channels
        self.hidden_units = hidden_units
        
        self.fc = nn.Sequential(
            nn.Linear(pose_channels*2, hidden_units),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pose1, pose2):
        bs = pose1.size(0)    
        comb_pose = torch.cat([pose1, pose2], dim=1)
        comb_pose = comb_pose.view(-1, self.pose_channels*2)

        return self.fc(comb_pose)  # bs*h*w x 1

class LSTMPoseGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layers = layers

        self.embedding = nn.Linear(input_size, hidden_size)

        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for _ in range(layers)]
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

        self.h, self.c = self.init_hidden()

    def init_hidden(self):
        h = []
        c = []
        for i in range(self.layers):
            h.append(Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda())
            c.append(Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda())

        return h, c

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        emb = self.embedding(x)

        for i in range(self.layers):
            self.h[i], self.c[i] = self.lstm[i](emb, (self.h[i], self.c[i]))
            emb = self.h[i]
        
        out = self.fc(emb)
        return out

# testing code
if __name__ == '__main__':
    batch_size=2
    channels=3
    height=128
    width=128

    torch.cuda.set_device(0)
    net = Encoder(3,128).cuda()
    inp = Variable(torch.FloatTensor(batch_size, channels, height, width)).cuda()
    out, skip = net(inp)
    print ('content after encoder: ', out.shape)

    net = Encoder(3,10,normalize=True).cuda()
    pose, _ = net(inp)
    print ('pose after encoder: ', pose.shape)

    net = Decoder(128+10,3,use_skip=True).cuda()
    out = net(out, skip, pose)
    print ('out after decoder: ', out.shape)

    net = SceneDiscriminator(10,100).cuda()
    pose1 = Variable(torch.FloatTensor(batch_size,10,5,5)).cuda()
    pose2 = Variable(torch.FloatTensor(batch_size,10,5,5)).cuda()
    out = net(pose1, pose2)
    print ('out after SceneDiscriminator: ', out.shape)

    net = LSTMPoseGenerator(128+5,256,10,batch_size).cuda()
    inp = Variable(torch.FloatTensor(batch_size,128+5)).cuda()
    out = net(inp)
    print ('out after LSTMPoseGenerator: ', out.shape)