import torch.nn as nn


# cnn input para:
'''
conv1 = []:
[0]conv1_input_channels;
[1]conv1_output_channels;
[2]conv1_kernel_size;
[3]conv1_stride;
[4]conv1_padding; (kernel_size-1)/2


conv2 = []:
[0]conv2_input_channels;
[1]conv2_output_channels;
[2]conv2_kernel_size;
[3]conv2_stride;
[4]conv2_padding; (kernel_size-1)/2


linear:
[0]input dim
[1]output dim
'''

class CNN(nn.Module):
    def __init__(self, conv1_info, conv2_info, linear_info):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_info[0],
                      conv1_info[1],
                      conv1_info[2],
                      conv1_info[3],
                      conv1_info[4]),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv2_info[0],
                      conv2_info[1],
                      conv2_info[2],
                      conv2_info[3],
                      conv2_info[4]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(linear_info[0], linear_info[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output