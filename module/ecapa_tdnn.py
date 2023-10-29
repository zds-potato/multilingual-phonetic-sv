import math
import torch
import torch.nn as nn

try:
    from ._pooling import *
except:
    from _pooling import *


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class ECAPA_TDNN(nn.Module):

    def __init__(self, n_mels=80, embedding_dim=192, channel=512, pooling_type="ASP"):

        super(ECAPA_TDNN, self).__init__()

        self.conv1  = nn.Conv1d(80, channel, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(channel)
        self.layer1 = Bottle2neck(channel, channel, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channel, channel, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channel, channel, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3*channel, 3*channel, kernel_size=1)

        if pooling_type == "Temporal_Average_Pooling" or pooling_type == "TAP":
            self.pooling = Temporal_Average_Pooling()
            self.bn2 = nn.BatchNorm1d(3*channel)
            self.fc = nn.Linear(3*channel, embedding_dim)
            self.bn3 = nn.BatchNorm1d(embedding_dim)


        elif pooling_type == "Temporal_Statistics_Pooling" or pooling_type == "TSP":
            self.pooling = Temporal_Statistics_Pooling()
            self.bn2 = nn.BatchNorm1d(3*channel*2)
            self.fc = nn.Linear(3*channel*2, embedding_dim)
            self.bn3 = nn.BatchNorm1d(embedding_dim)

        elif pooling_type == "Self_Attentive_Pooling" or pooling_type == "SAP":
            self.pooling = Self_Attentive_Pooling(3*channel)
            self.bn2 = nn.BatchNorm1d(3*channel)
            self.fc = nn.Linear(3*channel, embedding_dim)
            self.bn3 = nn.BatchNorm1d(embedding_dim)

        elif pooling_type == "Attentive_Statistics_Pooling" or pooling_type == "ASP":
            self.pooling = Attentive_Statistics_Pooling(3*channel)
            self.bn2 = nn.BatchNorm1d(3*channel*2)
            self.fc = nn.Linear(3*channel*2, embedding_dim)
            self.bn3 = nn.BatchNorm1d(embedding_dim)

        else:
            raise ValueError(
                '{} pooling type is not defined'.format(pooling_type))

        print("resnet num_channels: {}".format(channel))
        print("n_mels: {}".format(n_mels))
        print("embedding_dim: {}".format(embedding_dim))
        print("pooling_type: {}".format(pooling_type))


    def forward(self, x):
        x = x.squeeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1)) #ecapa_tdnn_large: batch * 3072 * 202
        x = self.relu(x)

        x=self.pooling(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn3(x)

        return x
 
def ecapa_tdnn(n_mels=80, embedding_dim=192, channel=512, pooling_type="ASP"):
    model = ECAPA_TDNN(n_mels=n_mels, embedding_dim=embedding_dim, channel=channel, pooling_type=pooling_type)
    return model

def ecapa_tdnn_large(n_mels=80, embedding_dim=192, channel=1024, pooling_type="ASP"):
    model = ECAPA_TDNN(n_mels=n_mels, embedding_dim=embedding_dim, channel=channel, pooling_type=pooling_type)
    return model


