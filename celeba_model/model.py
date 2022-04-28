from torch import nn
from torch.optim import Adam

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
vlb_scale_factor = 128 ** 2

def MLPBlock(dim):
    return SkipConnection(
        nn.BatchNorm2d(dim),
        nn.LeakyReLU(),
        nn.Conv2d(dim, dim, 1)
    )

# Corresponds to the proposal when not all data is available.
proposal_network = nn.Sequential(
    nn.Conv2d(6, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.AvgPool2d(2, 2), # 64 x 64
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1), # 32 x 32
    ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1), # 16 x 16
    ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1), # 8 x 8
    ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1), # 4 x 4
    ResBlock(128, 64), ResBlock(128, 64),
    ResBlock(128, 64), ResBlock(128, 64),
    nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1), # 2 x 2
    ResBlock(256, 128), ResBlock(256, 128),
    ResBlock(256, 128), ResBlock(256, 128),
    nn.AvgPool2d(2, 2), nn.Conv2d(256, 512, 1), # 1 x 1
    MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
)


# Corresponds to psi, take as input both x and b hence the size 6 as input Basically give z.
prior_network = nn.Sequential(
    MemoryLayer('#0'),
    nn.Conv2d(6, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    MemoryLayer('#1'),
    nn.AvgPool2d(2, 2), # 64 x 64
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    MemoryLayer('#2'),
    nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1), # 32 x 32
    ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    MemoryLayer('#3'),
    nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1), # 16 x 16
    ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    MemoryLayer('#4'),
    nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1), # 8 x 8
    ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    MemoryLayer('#5'),
    nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1), # 4 x 4
    ResBlock(128, 64), ResBlock(128, 64),
    ResBlock(128, 64), ResBlock(128, 64),
    MemoryLayer('#6'),
    nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1), # 2 x 2
    ResBlock(256, 128), ResBlock(256, 128),
    ResBlock(256, 128), ResBlock(256, 128),
    MemoryLayer('#7'),
    nn.AvgPool2d(2, 2), nn.Conv2d(256, 512, 1), # 1 x 1
    MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
)

# Corresponds to the decoder network phi, takes as input the latent representation which is a vector of 256 channels and dim 1x1. It is divided by two because we need to parametrized the distribution in mean and sigmas.
generative_network = nn.Sequential(
    nn.Conv2d(256, 256, 1),
    MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
    nn.Conv2d(256, 128, 1), nn.Upsample(scale_factor=2),
    MemoryLayer('#7', True), nn.Conv2d(384, 128, 1),
    ResBlock(128, 64), ResBlock(128, 64),
    ResBlock(128, 64), ResBlock(128, 64),
    nn.Conv2d(128, 64, 1), nn.Upsample(scale_factor=2),
    MemoryLayer('#6', True), nn.Conv2d(192, 64, 1),
    ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    nn.Conv2d(64, 32, 1), nn.Upsample(scale_factor=2),
    MemoryLayer('#5', True), nn.Conv2d(96, 32, 1),
    ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
    MemoryLayer('#4', True), nn.Conv2d(48, 16, 1),
    ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2),
    MemoryLayer('#3', True), nn.Conv2d(24, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.Upsample(scale_factor=2),
    MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.Upsample(scale_factor=2),
    MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    MemoryLayer('#0', True), nn.Conv2d(14, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.Conv2d(8, 6, 1),
)
