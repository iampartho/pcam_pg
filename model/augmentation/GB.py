# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:29:59 2020

@author: S. M. Jawwad Hossain
"""
import math
import torch
import torch.nn as nn
from torch.nn.functional import conv2d



#The kernel_size, var_x, var_y all should be torch_tensors
def gaussian_kernel(kernel_size, var_x, var_y):
    ax = torch.round(torch.linspace(-math.floor(kernel_size/2), math.floor(kernel_size/2),
                                    kernel_size), out=torch.FloatTensor())
    x = ax.view(1, -1).repeat(ax.size(0), 1).to('cuda')
    y = ax.view(-1, 1).repeat(1, ax.size(0)).to('cuda')
    x2 = torch.pow(x,2)
    y2 = torch.pow(y,2)
    std_x = torch.pow(var_x, 0.5)
    std_y = torch.pow(var_y, 0.5)
    temp = - ((x2/var_x) + (y2/var_y)) / 2
    kernel = torch.exp(temp)/(2*math.pi*std_x*std_y)
    kernel = kernel/kernel.sum()
    return kernel

# 1, 1
    
class GB2d(nn.Module):
    def __init__(self, in_channel, kernel_size, var_x = None, var_y = None, aug_train = True, padding = 0):
        super(GB2d, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.aug_train = aug_train
        
        if self.aug_train:
            self.var_x = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.var_y = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.init_parameters()
        else:
            self.var_x = torch.FloatTensor([var_x]).to('cuda')
            self.var_y = torch.FloatTensor([var_y]).to('cuda')
            self.kernel = gaussian_kernel(self.kernel_size, self.var_x, self.var_y)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        
            
    def init_parameters(self):
        self.var_x.data.uniform_(0, 1)
        self.var_y.data.uniform_(0, 1)
        
    def forward(self, input):
        input_shape = input.shape
        batch_size, h, w = input_shape[0], input_shape[2], input_shape[3]
        
        if self.aug_train:
            self.kernel = gaussian_kernel(self.kernel_size, self.var_x, self.var_y)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            
        output = conv2d(input.view(batch_size*self.in_channel,1,h,w), self.kernel, padding = self.padding)
        h_out = math.floor((h - self.kernel_size + 2*self.padding)+1)
        w_out = math.floor((w - self.kernel_size + 2*self.padding)+1)
        output = output.view(batch_size, self.in_channel, h_out, w_out)
        return output
    
    
    
    
    
    
    
    
    
    
    