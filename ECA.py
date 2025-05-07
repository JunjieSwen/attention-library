import torch
import math
import torch.nn as nn


class ECA_net(nn.Module):
    def __init__(self, in_channel, b=1, gama=2):
        super(ECA_net, self).__init__()
        kernel_size = int(abs((math.log(in_channel,2)+b)/gama))
        kernel = kernel_size if kernel_size % 2 else kernel_size+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel,1,padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        b,c,w,h = x.shape
        y = self.avg_pool(x)
        # (b,c,1,1)->(b,1,c)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)



if __name__ ==  '__main__':
    x = torch.randn([1,3,16,16])
    eca_net = ECA_net(x.shape[1])
    output = eca_net(x)
    print(output.shape)
