import torch
import torch.nn as nn
import torchvision.models as models

# VGG feature extractor
class VGGFeatureExtractor(nn.Module):
    """
        Extracts features from VGG19 at relu3_3 (layer index 18).
        Kept it frozen, used only for perceptual loss computation
    """
    def __init__(self):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # take only upto relu3_3
        self.features = nn.Sequential(
            *list(vgg.features.children())[:18]
        )
        # freeze
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)
    

# residual block
class ResidualBlock(nn.Module):
    """
        Conv -> BN -> PReLU -> Conv -> BN -> skip conn
    """
    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(), # better than LeakyReLU for skip conns
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x) # skip conn

# upsample block
class UpsampleBlock(nn.Module):
    """
        Conv -> PixelShuffle(r=2) -> PReLU
        Doubles spatial dims: (B, C, H, W) -> (B, C, H*2, W*2)
        Conv must output C * r^2 = C * 4 channels to work
    """
    def __init__(self, channels=64, upscale=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale), # (B, C*4, H, W) -> (B, C, H*2, W*2)
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

# G
class Generator(nn.Module):
    """
        Input : LR image (B, 3, 16, 16)
        Output: HR image (B, 3, 64, 64)
        4x upscaling via 2x pixelshuffle blocks
    """

    def __init__(self, num_res_blocks=16, channels=64):
        super().__init__()

        # initial feat extraction from LR image
        self.initial = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=9, stride=1, padding=4), # 9x9 large kernel for large receptive field - output feat map preserves 
            # (N - K + 2P)/S + 1
            nn.PReLU()
        )

        # 16 res blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # post residual conv + BN - before final skip
        self.post_res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # two upsample blocks: 16 -> 32 -> 64
        self.upsample = nn.Sequential(
            UpsampleBlock(channels, upscale=2),
            UpsampleBlock(channels, upscale=2)
        )

        # output : RGB
        self.final = nn.Conv2d(channels, 3, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        initial = self.initial(x) # (B, 64, 16, 16)
        res = self.res_blocks(initial) # (B, 64, 16, 16)
        post = self.post_res(res) # (B, 64, 16, 16)
        fused = initial + post # gloabl skip -- across ALL res block
        up = self.upsample(fused) # (B, 64, 64, 64)
        return self.tanh(self.final(up)) # (B, 3, 64, 64)


# D
class Discriminator(nn.Module):
    """
        VGG-style classifier/
        Input: (B, 3, 64, 64) HR image - real or G's output
        Output: (B, 1) - real/fake probability
    """

    def __init__(self, features_d=64):
        super().__init__()

        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            # no BN on first layer
            nn.Conv2d(3, features_d, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            block(features_d, features_d, stride=2), # 64 -> 32
            block(features_d, features_d * 2, stride=1),
            block(features_d * 2, features_d * 2, stride=2), # 32 -> 16
            block(features_d * 2, features_d * 4, stride=1),
            block(features_d * 4, features_d * 4, stride=2), # 16 -> 8
            block(features_d * 4, features_d * 8, stride=1),
            block(features_d * 8, features_d * 8, stride=2), # 8 -> 4

            nn.AdaptiveAvgPool2d(1), # (B, 512, 1, 1)
            nn.Flatten(), # (B, 512)
            nn.Linear(features_d * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# sanity check
if __name__ == "__main__":
    lr    = torch.randn(4, 3, 16, 16)
    hr    = torch.randn(4, 3, 64, 64)

    G     = Generator(num_res_blocks=16)
    D     = Discriminator()
    vgg   = VGGFeatureExtractor()

    fake_hr = G(lr)
    d_out   = D(fake_hr)
    f_real  = vgg(hr)
    f_fake  = vgg(fake_hr)

    print("G output:", fake_hr.shape)     # (4, 3, 64, 64)
    print("D output:", d_out.shape)       # (4, 1)
    print("VGG real features:", f_real.shape)   # (4, 256, 16, 16)
    print("VGG fake features:", f_fake.shape)   # (4, 256, 16, 16)