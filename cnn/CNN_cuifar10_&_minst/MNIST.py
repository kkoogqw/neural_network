#%%  lib
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

#%% define
torch.manual_seed(1)
#%% 获取训练集dataset
# train data
train_data = torchvision.datasets.MNIST(root='./mnist/',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=100,
                               shuffle=True)

# test data
test_data = torchvision.datasets.MNIST(root='./mnist/',
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())
# 取前全部10000个测试集样本
test_images = Variable(torch.unsqueeze(test_data.test_data, dim=1).float(), requires_grad=False)
test_labels = test_data.test_labels

#%% 打印MNIST数据集的训练集及测试集的尺寸
print("train data size:")
print(train_data.train_data.size())
print(train_data.train_labels.size())

#%% 打印MNIST测试集数据大小
print("test data size:")
print(test_data.test_data.size())
print(test_data.test_labels.size())

#%% 随机显示一个训练集图
# plt.imshow(train_data.train_data[3000].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[3000])
# plt.show()

#%% MNIST 相关参数定义
# image size: height = 1; size = 28 * 28
conv1_info = [1, 16, 5, 1, 2]
conv2_info = [16, 32, 5, 1, 2]
linear_info = [32 * 7 * 7, 10]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # in (height, size(n*n))=>(1,28,28)
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2
                      ),  # out (height, size(n*n))=>(1,28,28)->(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,28/2,28/2)-> (16,14,14)
        )
        self.conv2 = nn.Sequential(  # in => (16,14,14)
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2
                      ),  # out => (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # out => (32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 将（batch，32,7,7）->（batch，32*7*7）
        output = self.out(x)
        return output


#%% 定义CNN网络实例
cnn = CNN()

# optimizer = torch.optim.RMSprop(cnn.parameters(), lr=LR, alpha=0.9)
# optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

loss_function = nn.CrossEntropyLoss()

#%% 训练过程
EPOCH = 20
# GPU 测试
if_use_gpu = False
if torch.cuda.is_available():
    if_use_gpu = True

if if_use_gpu == True:
    cnn = cnn.cuda()

train_loss_record = []
for lap in range(EPOCH):
    start = time.clock()
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x, requires_grad=False)
        batch_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        output = cnn(batch_x)
        loss = loss_function(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每批次训练数量为100， 一共有60000/100 = 600 次
        # 每个十步输出一次,一共输出600/10=60条状态消息
        if step % 100 == 0:
            train_loss_record.append(loss.item())
            print("lap count: ", lap, "\tStep: ", step, "\ttrain loss: ", loss.item())
    end = time.clock()
    time_cost = end - start
    print("Train time cost:", time_cost, "s")

#%% 绘制loss 曲线
plt.figure(1)
plt.plot(train_loss_record)
plt.title("Train Loss Record")
plt.show()

#%% 测试结果
# 转移到CPU计算
cnn = cnn.cpu()
test_output = cnn(test_images)
predict_result = torch.max(test_output, 1)[1].data.squeeze()
#%%
test_result = predict_result.numpy()
label_result = test_labels.numpy()
# 误差计算
error_count = 0
error_record = []
for i in range(len(test_result)):
    if test_result[i] != test_labels[i]:
        error_count += 1
        error_record.append(i)
print("Correct Rate: ", 1 - (error_count/len(test_labels)))
print("Error predict count: ", error_count)
# print(error_record)
# print(test_result[error_record])
# print(label_result[error_record])

#%% 错误结果显示： 序号/预测的错误结果/真是结果
print("Error List:")
for i in range(len(error_record)):
    print("Error index in test-set: ", error_record[i],
          "predict label: ", test_result[error_record[i]],
          "True labels:", label_result[error_record[i]])
# accuracy = sum(pred_y == test_y) / test_y.size(0)

#%% 抽取前十个错误样例显示图片
view_images = test_data.test_data[:10000].view(-1, 28*28).type(torch.FloatTensor)/255.
figure, pic = plt.subplots(1, 15, figsize=(15, 2))
plt.ion()
for i in range(15):
    pic[i].imshow(np.reshape(test_images.data.numpy()[error_record[i]], (28, 28)), cmap="gray")
    pic[i].set_xticks(())
    pic[i].set_yticks(())
    pic[i].set_title("predict:" + str(test_result[error_record[i]]) +
                     "\nlabel:" + str(label_result[error_record[i]]))
plt.draw()