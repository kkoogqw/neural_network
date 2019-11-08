#%%
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import AutoEncoder as encoder

# from the mnist dataset import the trian data
# 载入数据 训练集/测试集
train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), )
train_data_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)
# from the mnist dataset import the test data
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(), )
test_loader = Data.DataLoader(dataset=test_data, batch_size=100, shuffle=True)

#%%
print(train_data.train_data, train_data.train_data.size())
print(train_data.train_labels, train_data.train_labels.size())
#%%
print(test_data.test_data, test_data.test_data.size())
print(test_data.test_labels, test_data.test_labels.size())

#%% build an auto-encoder and classifier use the class defined in "AutoEncoder.py"
# 建立编码机/分类器模型
auto_encoder = encoder.AutoEncoder()
classifier = encoder.Classifier(n_feature=2, n_hidden=20, n_output=10)

# using the Adam() & MSE to do the BP for encoder
# 优化函数与误差分析
encoder_optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=0.001)
encoder_loss_rate_func = nn.MSELoss()
# using the
classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001)
classifier_loss_rate_func = nn.CrossEntropyLoss()
#%%
# 定义编码机/分类器的训练次数
encoder_EPOCH_count = 50
classifier_EPOCH_count = 100

#%% train the auto-encoder
# 编码机训练模型
encoder_train_loss_count = []
for lap in range(encoder_EPOCH_count):
    for step, (x, b_lable) in enumerate(train_data_loader):
        b_x = x.view(-1, 28 * 28)  # (28*28)
        b_y = x.view(-1, 28 * 28)
        # train the auto-encoder network
        encoded, decoded = auto_encoder(b_x)
        encoder_loss = encoder_loss_rate_func(decoded, b_y)  # mean square error
        encoder_optimizer.zero_grad()                    # clear gradients for this training step
        encoder_loss.backward()                          # backpropagation, compute gradients
        encoder_optimizer.step()                         # apply gradients

        loss_record = encoder_loss.data.numpy()
        encoder_train_loss_count.append(loss_record)
        print("AutoEncoder train lap count:", lap, "step count:", step, "encode loss:", loss_record)

encoder_train_loss_count = np.array(encoder_train_loss_count)
encoder_loss_mean = encoder_train_loss_count.mean()
print(encoder_loss_mean)


#%% train the classifier
# 训练分类器
classifier_train_loss_count = []
for lap in range(classifier_EPOCH_count):
    for step, (x, b_lable) in enumerate(train_data_loader):
        b_x = x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
        encoded, decoded = auto_encoder(b_x)
        # train the classifier network
        classifier_out = classifier(encoded)
        classifier_loss = classifier_loss_rate_func(classifier_out, b_lable)
        classifier_optimizer.zero_grad()
        classifier_loss.backward()
        classifier_optimizer.step()

        loss_record = classifier_loss.data.numpy()
        classifier_train_loss_count.append(loss_record)
        print("Classifier train lap count:", lap, "step count:", step, "loss:", loss_record)

classifier_train_loss_count = np.array(classifier_train_loss_count)
classifier_loss_mean = classifier_train_loss_count.mean()
print(classifier_loss_mean)

#%% train procession data
# 绘图训练过程的误差下降曲线
plt.figure(1)
plt.plot(encoder_train_loss_count)
plt.title("Auto encoder Loss record")
plt.xlabel("Trian laps")
plt.ylabel("Loss Rate")
plt.show()

plt.figure(2)
plt.plot(classifier_train_loss_count)
plt.title("Classifier Loss record")
plt.xlabel("Trian laps")
plt.ylabel("Loss Rate")
plt.show()



'''SVM 分类器'''
#%%
from sklearn import cross_validation
from sklearn.svm import SVC
import sklearn.svm as svm

# get data for svm

svm_trian_data, _ = auto_encoder(train_data.train_data[:10000].view(-1, 28*28).type(torch.FloatTensor)/255.)
svm_trian_label = train_data.train_labels.numpy()

print(svm_trian_data[0])
#%%
svm_classifier = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
svm_classifier.fit(svm_trian_data, svm_trian_label)


'''下面是对全部的测试集数据进行编码分类处理，根据编码之后的结果进行分类，查看分类效率'''

#%%  use test data to run the auto encoder:
# the test-dataset has 10,000 examples:
# 获取测试集数据，送入编码机编码
test_input_data = test_data.test_data[:10000].view(-1, 28*28).type(torch.FloatTensor)/255.
test_encoded, test_decoded = auto_encoder(test_input_data)

#%% use the classifier to do classify!
# 编码结果分类
classify_result = classifier(test_encoded)
# get the max value in every unit
classifier_prediction = torch.max(classify_result,1)[1]
predy = classifier_prediction.data.numpy().squeeze()

# 计算分类结果的误差
data_label = test_data.test_labels[:10000].numpy()
print(data_label)
print(predy)
error_count = 0
for i in range(len(predy)):
    if predy[i] != data_label[i]:
        error_count += 1
print("Error Rate:", error_count/len(predy))

'''下面是提取测试集中的500个样本数据进行可视化处理，分别展示：
    1.编码-解码之后与原图的效果
    2.编码之后的图像对应的标签关系
    3.编码-分类之后图像与分类结果的关系
'''
#%% draw the picture of before encode and after decode
# get the data of 500 sample in dataset
view_data = test_data.test_data[:500].view(-1, 28*28).type(torch.FloatTensor)/255.

figure, pic = plt.subplots(2, 10, figsize=(10, 4))
plt.ion()
for i in range(10):
    pic[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)))
    pic[0][i].set_xticks(())
    pic[0][i].set_yticks(())
for i in range(10):
    pic[1][i].clear()
    pic[1][i].imshow(np.reshape(test_decoded.data.numpy()[i], (28, 28)))
    pic[1][i].set_xticks(())
    pic[1][i].set_yticks(())
plt.draw()

#%% use the dataset label to show the after-encoded picture(only choose 500 samples)
show_encoded, _ = auto_encoder(view_data)
show_classify = classifier(show_encoded)
show_classify_label = torch.max(classify_result, 1)[1]
show_classify_label = show_classify_label.data.numpy().squeeze()


#%% show the encode result with labels:
plt.figure(4)
plt.ion()
plt.cla()
X, Y = show_encoded.data[:, 0].numpy(),  show_encoded.data[:, 1].numpy()
show_labels = test_data.test_labels[:500].numpy()
for x, y, s in zip(X, Y, show_labels):
    c = cm.rainbow(int(255 * s / 9))
    plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('After encoding with labels')
    plt.show()

#%% show data with labels after classified:
plt.figure(5)
plt.ion()
plt.cla()
X, Y = show_encoded.data[:, 0].numpy(),  show_encoded.data[:, 1].numpy()

for x, y, s in zip(X, Y, show_classify_label):
    c = cm.rainbow(int(255 * s / 9))
    plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('After classified with labels')
    plt.show()
