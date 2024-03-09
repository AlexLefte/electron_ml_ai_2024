from torch import nn


class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        # Initialize the layers
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights and biases of the U-Net constituent blocks
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier uniform: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.00001)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)  # Initialize scale to 1
                nn.init.constant_(module.bias, 0)  # Initialize shift to 0

    def forward(self, x):
        out = self.dncnn(x)
        return out
