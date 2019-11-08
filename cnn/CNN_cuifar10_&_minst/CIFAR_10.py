#author: yuquanle
#date: 2018.2.5
#Classifier use PyTorch (CIFAR10 dataset)

#%%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import  Variable

#%%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(root='./cifar_10',
                                        train=True,
                                        transform=transform)
train_loader = Data.DataLoader(train_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=0)

test_data = torchvision.datasets.CIFAR10(root='./cifar_10',
                                       train=False,
                                       transform=transform)
test_loader = Data.DataLoader(test_data,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)

# 3*32*32

#%%

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # in (height, size(n*n))=>(1,32,32)
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=2
                      ),  # out (height, size(n*n))=>(3,32,32)->(16,32,32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,32/2,32/2)-> (16,16,16)
        )
        self.conv2 = nn.Sequential(  # in => (16,16,16)
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2
                      ),  # out => (32,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2)  # out => (32,8,8)
        )
        self.out = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 将（batch，32,8,8）->（batch，32*8*8）
        output = self.out(x)
        return output


#%% our model
cnn = CNN()

loss_function = nn.CrossEntropyLoss()
# SGD with momentum
# optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(cnn.parameters(), lr=0.001)



#%%
if_use_gpu = False
if torch.cuda.is_available():
    if_use_gpu = True

if if_use_gpu == True:
    cnn = cnn.cuda()

train_loss_record = []
EPOCH = 50

for lap in range(EPOCH):
    start = time.clock()
    for step, data in enumerate(train_loader, 0):
        batch_x, batch_y = data
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)
        if if_use_gpu == True:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        output = cnn(batch_x)
        loss = loss_function(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            train_loss_record.append(loss.item())
            print("lap count: ", lap, "\tStep: ", step, "\ttrain loss: ", loss.item())
    end = time.clock()
    time_cost = end - start
    print("Train time cost:", time_cost, "s")

plt.figure(1)
plt.plot(train_loss_record)
plt.title("Train Loss Record")
plt.show()

#%% use test data for trained cnn

cnn = cnn.cpu()
prediction_result = np.array([])
true_labels = np.array([])
for test_input_batch in test_loader:
    test_images, test_labels = test_input_batch
    test_ouput = cnn(test_images)

    pred = torch.max(test_ouput.data, 1)[1]
    temp_prediction = pred.data.numpy().squeeze()
    temp_labels = test_labels.numpy()

    prediction_result = np.append(prediction_result, temp_prediction)
    true_labels = np.append(true_labels, temp_labels)

#%%
prediction_result.astype(int)
true_labels.astype(int)

print(prediction_result, len(prediction_result))
print(true_labels, len(true_labels))

error_count = 0
error_index_record = []

for i in range(len(prediction_result)):
    if int(prediction_result[i]) != int(true_labels[i]):
        error_count += 1
        error_index_record.append(i)

correct_rate = 1 - (error_count / len(prediction_result))
print("Correct Rate:", correct_rate)
print("Error prediction count: ", error_count)
print("Error List:")
#%%
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(error_index_record)
for i in range(len(error_index_record)):
    print("image index in test data set:", error_index_record[i],
          "\tpredict result:(number/class name)", (prediction_result[error_index_record[i]]),
                                                  classes[int(prediction_result[error_index_record[i]])],
          "\tReal result: (number/class name)", (true_labels[error_index_record[i]]),
                                                classes[int(true_labels[error_index_record[i]])])

#%% print images
# choose 10 error predict images to show
def imgshow(img, index):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("predict label:" + str(classes[int(prediction_result[error_index_record[i]])]) +
              "\ntrue label:" + str(classes[int(true_labels[error_index_record[i]])]))
    plt.show()

# show images

for i in range(10):
    plt.figure()
    images, labels = test_data[error_index_record[i]]
    imgshow(torchvision.utils.make_grid(images), i)
# print labels

