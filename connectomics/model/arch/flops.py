    
import torch
from torch import nn
from thop import profile
from GBPNet import GBPNet



if __name__ == '__main__':
    
    
    net = GBPNet()
    # input = torch.randn(44694, 3, 81, 81, 81)  # 1191423.36134016
    input = torch.randn(1, 3, 81, 81, 81)  # FLOPs = 26.65734464G
    flops, params = profile(net, inputs=(input,))

    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')