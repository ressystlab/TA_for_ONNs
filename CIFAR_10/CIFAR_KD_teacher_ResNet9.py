# Train the ResNet-9 teacher
# Code adapted from https://github.com/vvs07/noisyphotonicDNNs.git

# Created: Aug 29, 2025
# Last Update: Sep 1, 2025
# Author: Xuening D

import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

#Resnet 9 architecture
class Net(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1,bias = False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.res1 = nn.Sequential(conv_block(128, 128),conv_block(128, 128))
        self.res1conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.res1batchnorm1 = nn.BatchNorm2d(128)
        self.res1conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.res1batchnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.res2 = nn.Sequential(conv_block(512, 512),conv_block(512, 512))
        self.res2conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.res2batchnorm1 = nn.BatchNorm2d(512)
        self.res2conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.res2batchnorm2 = nn.BatchNorm2d(512)
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))


    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = (x-torch.mean(x))/(torch.std(x))
        y = x
        x = F.relu(self.res1batchnorm1(self.res1conv1(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = F.relu(self.res1batchnorm2(self.res1conv2(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = x+y
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        x = (x-torch.mean(x))/(torch.std(x))
        x = self.pool(F.relu(self.batchnorm4(self.conv4(x))))
        x = (x-torch.mean(x))/(torch.std(x))
        z = x
        x = F.relu(self.res2batchnorm1(self.res2conv1(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = F.relu(self.res2batchnorm2(self.res2conv2(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = x+z
        x = self.classifier(x)
        return x
    
    # extract the logits from the convolutional layer
    def forward_wo_classifier(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = (x-torch.mean(x))/(torch.std(x))
        y = x
        x = F.relu(self.res1batchnorm1(self.res1conv1(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = F.relu(self.res1batchnorm2(self.res1conv2(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = x+y
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        x = (x-torch.mean(x))/(torch.std(x))
        x = self.pool(F.relu(self.batchnorm4(self.conv4(x))))
        x = (x-torch.mean(x))/(torch.std(x))
        z = x
        x = F.relu(self.res2batchnorm1(self.res2conv1(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = F.relu(self.res2batchnorm2(self.res2conv2(x)))
        x = (x-torch.mean(x))/(torch.std(x))
        x = x+z
        return x


if __name__ == "__main__":
    stats = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2436, 0.2617))
    train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=(0, 10)),
                            transforms.ToTensor(),
                            transforms.Normalize(*stats,inplace=True)])
    valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    trainset = torchvision.datasets.CIFAR10(root='results/data', train=True,
                                            download=True, transform=train_tfms)
    testset = torchvision.datasets.CIFAR10(root='results/data', train=False,
                                        download=True, transform=valid_tfms)


    batch_size = 400
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=1,pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(testset, batch_size=batch_size*2,
                                            shuffle=False, num_workers=1,pin_memory=True)

    net2 = Net(3, 10)

    device = 'cpu'
    # define save path
    save_dir = "CIFAR_10/teachers"
    os.makedirs(save_dir, exist_ok=True)

    net2.load_state_dict(torch.load(f"{save_dir}/teacher_resnet9.pth", map_location=device))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_dl:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            #Inferencing on clean hardware
            outputs = net2(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test accuracy = {100 * correct / total:.2f} [%]')

    

    # save training set logits
    logit_save_path = f"{save_dir}/cifar_train_logits_resnet9_conv_train.npy"
    all_logits = []

    with torch.no_grad():
        for imgs, _ in train_dl:
            imgs = imgs.to(device)
            logits = net2.forward_wo_classifier(imgs)
            all_logits.append(logits.cpu())

    logits_tensor = torch.cat(all_logits)
    print(logits_tensor.shape)
    np.save(logit_save_path, logits_tensor.numpy())
    print(f"[Done] Logits saved at {logit_save_path}")

    # save test set logits
    logit_save_path = f"{save_dir}/cifar_train_logits_resnet9_conv_test.npy"
    all_logits = []

    with torch.no_grad():
        for imgs, _ in valid_dl:
            imgs = imgs.to(device)
            logits = net2.forward_wo_classifier(imgs)
            all_logits.append(logits.cpu())

    logits_tensor = torch.cat(all_logits)
    print(logits_tensor.shape)
    np.save(logit_save_path, logits_tensor.numpy())
    print(f"[Done] Logits saved at {logit_save_path}")