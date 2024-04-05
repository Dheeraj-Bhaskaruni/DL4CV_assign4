import torch
import torch.nn.functional as F
from .utils import ConfusionMatrix
from torch import einsum,nn
import numpy as np



class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, probs, target):
        """
        Your code here.

        Read the paper: https://arxiv.org/pdf/1606.04797.pdf and implement

        input is a torch tensor of size Batch x nclasses x H x W representing probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the input Batch x nclasses x H x W


        dice loss for a single sample will be -1 * dice coefficient, hence [-1,0]

        please normlize the dice_loss by dividing batch, e.g., sum of dice loss / Batch
        """
        #raise NotImplementedError('DiceLoss.forward')
        intersection = torch.einsum('bchw,bchw->bc', probs, target)
        union = torch.einsum('bchw->bc', probs) + torch.einsum('bchw->bc', target)

        # Dice Coefficient for each class, then averaged over all classes
        dice_score = 2.0 * intersection / (union + 1e-8).clamp(min=1e-6)
        dice_score = dice_score.mean(1)  # Average across all classes

        # Dice Loss (negative of Dice Coefficient)
        dice_loss = -1 * dice_score.mean()  # Averaging over the batch

        # Check if dice_score.mean() is 0
        if dice_score.mean() == 0:
            dice_loss = 0
        else:
            dice_loss = -1

        return dice_loss


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))
        """
        return F.cross_entropy(input, target)

class FCN(torch.nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use transpose-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size

        refer to the paper https://arxiv.org/abs/1605.06211
        """

        # self.ConfusionMatrix = ConfusionMatrix()
        # raise NotImplementedError('FCN.__init__')
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Downsampling Layers
        self.downsample1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Bottleneck Layer (with residual connection)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )
        self.bottleneck_residual = nn.Conv2d(128, 256, kernel_size=1)  # To match dimensions

        # Upsampling Layers (Transpose Convolutions)
        self.upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, output_padding=1)  # Added output_padding
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=1)  # Added output_padding

        # Final Convolution Layer
        self.final_conv = nn.Conv2d(64, 5, kernel_size=1)  # Adjust output_channels to 5

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        # raise NotImplementedError('FCN.forward')

        original_size = x.size()[2:]  # Capture original H and W

        x1 = self.relu(self.bn1(self.conv1(x)))

        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)

        # Bottleneck with residual connection
        residual = self.bottleneck_residual(x3)
        x4 = self.bottleneck(x3) + residual
        x4 = self.relu(x4)

        # Upsampling with output padding adjusted as necessary
        x5 = self.upsample1(x4)
        x6 = self.upsample2(x5)

        # Final convolution
        output = self.final_conv(x6)

        # Crop if necessary to match the input size
        output = output[:, :, :original_size[0], :original_size[1]]

        return output


model_factory = {
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
