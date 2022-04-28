from torch import nn
from torch.optim import Adam

import sys
import os
current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_file_path))
from mask_generators import ImageMaskGenerator
from nn_utils import ResBlock, MemoryLayer, SkipConnection
from prob_utils import normal_parse_params, GaussianLoss




# sampler from the model generative distribution
# here we return mean of the Gaussian to avoid white noise
def sampler(params):
    return normal_parse_params(params).mean


def optimizer(parameters):
    return Adam(parameters, lr=2e-4)


batch_size = 16

reconstruction_log_prob = GaussianLoss()

mask_generator = ImageMaskGenerator()

# improve train computational stability by dividing the loss
# by this scale factor right before backpropagation
vlb_scale_factor = 28 ** 2 # In the original paper, they divide by 128 ** 2 ie the number of pixel in the image, I just did the same thing

def MLPBlock(dim):
    return SkipConnection(
        nn.BatchNorm2d(dim),
        nn.LeakyReLU(),
        nn.Conv2d(dim, dim, 1)
    )

class Reshape(nn.Module):
    def __init__(self,shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        aux_x = x.reshape(self.shape)
        return aux_x

class NonSymetric0Pad(nn.Module):
    def __init__(self, shape):
        super(NonSymetric0Pad, self).__init__()
        self.shape = shape

    def forward(self, x):
        return nn.functional.pad(x, self.shape, mode='constant', value=0)
        

proposal_network = nn.Sequential(
    NonSymetric0Pad((4, 4, 2, 2,)),
    nn.Conv2d(6, 8, 1), # 32 x 64
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.AvgPool2d(2, 2), # 16 x 32
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),  # 8 x 16
    ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1), # 4 x 8
    ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1), # 2 x 4
    ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1), # 1 x 2
    Reshape((-1, 256, 1, 1, )),
    MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
)

prior_network = nn.Sequential(
    NonSymetric0Pad((4, 4, 2, 2,)),
    MemoryLayer('#0'),
    nn.Conv2d(6, 8, 1), # 32 x 64
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    MemoryLayer('#1'),
    nn.AvgPool2d(2, 2), #16x 32
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    MemoryLayer('#2'),
    nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),  # 8 x 16
    ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    MemoryLayer('#3'),
    nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1), # 4 x 8
    ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    MemoryLayer('#4'),
    nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1), # 2 x 4
    ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    MemoryLayer('#5'),
    nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),# 1 x 2
    Reshape((-1, 256, 1, 1, )),
    MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
)



generative_network = nn.Sequential(
    nn.Conv2d(128, 128, 1), # 1 x 1
    MLPBlock(128), MLPBlock(128), MLPBlock(128), MLPBlock(128),
    Reshape((-1, 64, 1, 2,)),    # 1 x 2
    nn.Conv2d(64, 32, 1), nn.Upsample(scale_factor=2), # 2 x 4
    MemoryLayer('#5', True), nn.Conv2d(96, 32, 1),
    ResBlock(32, 16), ResBlock(32, 16),
    ResBlock(32, 16), ResBlock(32, 16),
    nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2), # 4 x 8
    MemoryLayer('#4', True), nn.Conv2d(48, 16, 1),
    ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2), # 8 x 16
    MemoryLayer('#3', True), nn.Conv2d(24, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.Conv2d(8, 8, 1), nn.Upsample(scale_factor=2), # 16 x 32
    MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.Conv2d(8, 8, 1), nn.Upsample(scale_factor=2), # 32 x 64
    MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    MemoryLayer('#0', True), nn.Conv2d(14, 8, 1), # 32 x 64
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.Conv2d(8, 6, 1),
    NonSymetric0Pad((-4, -4, -2, -2,)),
)

