import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class classInvariant(nn.Module):
    def __init__(self):
        super(classInvariant, self).__init__()
        self.cos = torch.nn.CosineEmbeddingLoss(margin=0, size_average=True)

    def computeC(self, a, b):
        result = np.squeeze(np.ones((len(a), 1)))
        for i in range(0, len(a)):
            if a[i] != b[i]:
                result[i] = -1

        return result

    def to_var(self, x, volatile=False):
        x = x.cuda()
        return Variable(x, volatile=volatile)

    def forward(self, mmd1, mmd2, label1, label2):
        cosine = torch.nn.CosineSimilarity(dim=0)
        if len(label1) == len(label2):
            return self.cos(mmd1.data, mmd2.data, self.to_var(torch.from_numpy(self.computeC(label1, label2))))
        else:
            result = 0
            num = 0.0
            # print(len(label1))
            # print(len(label2))
            # print(mmd1[1])
            # print(mmd2[1])
            for i in range(0, len(label1)):
                for j in range(0, len(label2)):
                    if label1[i] == label2[j]:
                        num += 1.0
                        result += cosine(mmd1[i], mmd2[j])
            if num != 0:
                result /= num
                # print(result)
                # return Variable(result, volatile=False)
                return result
            else:
                return self.to_var(torch.FloatTensor(1))


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    # print ("source")
    # print (source.shape)
    # print ("target")
    # print (target.shape)
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    # print ("mmdsource")
    # print (source.shape)
    # print ("mmdtarget")
    # print (target.shape)
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def CORAL(source, target):
    d = source.data.shape[1]
    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)
    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    return loss


class DeepCORAL(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        # self.fc0 = nn.Linear(4096, 256)
        # self.fc1 = nn.Linear(256, num_classes)

        self.fc1 = nn.Linear(4096, num_classes)
        # initialize according to CORAL paper experiment
        self.fc1.weight.data.normal_(0, 0.005)

    def forward(self, source):
        # source = self.fc0(source)
        # tmp_mmd = source
        # source = self.fc1(source)

        tmp_mmd = source
        source = self.fc1(source)

        return source, tmp_mmd


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.bottleneck(x)

        # residual = x
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        #
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        #
        # x = self.conv3(x)
        # x = self.bn3(x)
        #
        # x += residual
        # x = self.relu(x)

        feaure = x
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x, feaure


class DomainDis(nn.Module):
    def __init__(self):
        super(DomainDis, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc0 = nn.Linear(4096, 1024)
        self.fc1 = nn.Linear(1024, 1)
        self.si = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), 1024 * 2 * 2)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.si(x)
        return x


class concatNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(concatNet, self).__init__()
        self.fc0 = nn.Linear(2 * num_classes, num_classes)

    def forward(self, x, y):
        out = self.fc0(torch.cat((x, y), 1))

        return out
