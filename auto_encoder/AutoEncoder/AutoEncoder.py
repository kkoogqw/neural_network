import torch
import torch.nn as nn
import torch.nn.functional as F

# build the Auto encoder net work model
# 建立auto-encoder编码机模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 2),  # 28*28 -> 2
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),  # 压缩到(0, 1)区间内
        )

    def forward(self, input_data):
        encode_reuslt = self.encoder(input_data)
        decode_result = self.decoder(encode_reuslt)

        return encode_reuslt, decode_result

# build the classifier net work model
# 定义分类器模型
class Classifier(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Classifier, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input_data):
        input_data = F.relu(self.hidden(input_data))
        input_data = self.out(input_data)
        return input_data



