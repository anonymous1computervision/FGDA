from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torch.autograd import Variable
from torch.autograd import Function, Variable
from torch.autograd import grad
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import os
import time
import datetime
from PIL import Image
import utils
from data_loader import *
from tqdm import tqdm
import cv2
import math

CUDA = True if torch.cuda.is_available() else False

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from models import *

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


class mySolver(object):

    def __init__(self, config):
        if config['random'] == False:
            SEED = 0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)

        self.config = config

        self.CUDA = True
        self.BATCH_SIZE = config['batch_size']

        self.target_loader_test = get_dataloader_target(batch_size=self.BATCH_SIZE[1], domain=config['target'],
                                                        istrain='test')
        print (len(self.target_loader_test))


        self.build_model()

    def build_model(self):
        self.featureExactor = AlexNet()
        self.featureExactor.cuda()

        self.featureExactor1 = AlexNet()
        self.featureExactor1.cuda()
        # self.featureExactor = torch.nn.DataParallel(self.featureExactor)

        self.classfier = DeepCORAL(200)
        self.classfier.cuda()
        # self.classfier = torch.nn.DataParallel(self.classfier)

        self.domain = [1, 1]
        for i in range(2):
            self.domain[i] = DomainDis()
            self.domain[i].cuda()
            # self.classfier = torch.nn.DataParallel(self.classfier)

    def load_model(self, modeldir):
        self.featureExactor.load_state_dict(torch.load(modeldir + '/featureExactor_checkpoint.tar'))
        self.featureExactor1.load_state_dict(torch.load(modeldir + '/featureExactor1_checkpoint.tar'))
        self.classfier.load_state_dict(torch.load(modeldir + '/classifier_checkpoint.tar'))

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def cropBatch(self, batch, featuremap):
        # featuremap = self.cropNet(batch)
        featuremap = featuremap
        featuremap = np.sum(featuremap.data.cpu().numpy(), axis=1)

        newBatch = []

        for i in range(len(batch)):

            tmp_heat = featuremap[i]
            tmp_heat -= np.sum(tmp_heat) / 36
            tmp_heat = np.maximum(tmp_heat, 0)
            tmp_heat /= np.max(tmp_heat)
            tmp_heatmap = np.uint8(255 * tmp_heat)
            #
            # heatmap = cv2.resize(tmpss1, (224, 224))
            # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # name1 = '/unsullied/sharefs/wangyimu/rootfs-home/projects/finegrained/mymodel/pic/' + str(
            #     epoch) + '_' + str(batch_idx) + '_heat.jpg'
            # cv2.imwrite(name1, heatmap)
            _, binary = cv2.threshold(tmp_heatmap, 127, 255, cv2.THRESH_BINARY)

            _1, contours, _2 = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            tmp_max = -1
            tmp_maxi = -1
            for i in range(len(contours)):
                cnt = contours[i]
                _, _, w, h = cv2.boundingRect(cnt)
                if w * h > tmp_max:
                    tmp_max = w * h
                    tmp_maxi = i
            tmpx, tmpy, tmpw, tmph = cv2.boundingRect(contours[tmp_maxi])
            tmpx1, tmpy1, tmpx2, tmpy2 = int(tmpx * 227 / 6), int(tmpy * 227 / 6), int(
                math.ceil((tmpx + tmpw) * 227 / 6)), int(math.ceil((tmpy + tmph) * 227 / 6))

            # tmp_img = batch[i].data.cpu().numpy().transpose(1, 2, 0)
            tmp_img = batch[i].data.cpu().numpy()
            tmp_img = tmp_img.transpose(1, 2, 0)
            tmp_img = Image.fromarray(np.uint8(tmp_img))
            tmp_bbox = (tmpx1, tmpy1, tmpx2, tmpw)
            tmp_bbox = tuple(tmp_bbox)
            tmp_img = tmp_img.crop(tmp_bbox).resize((227, 227))
            tmpiimg = np.asarray(tmp_img)

            newBatch.append(tmpiimg)

            # name1 = '/unsullied/sharefs/wangyimu/rootfs-home/projects/finegrained/mymodel/pic/' + str(
            #     epoch) + '_' + str(batch_idx) + '_heat1.jpg'
            # cv2.imwrite(name1, heatmaps)
            #
            # tmpss0 = target_data.data.cpu().numpy()
            # tmpss0 = tmpss0[0].transpose(1, 2, 0)
            # tmpss0 = np.uint8(255 * tmpss0)
            # name0 = '/unsullied/sharefs/wangyimu/rootfs-home/projects/finegrained/mymodel/pic/' + str(
            #     epoch) + '_' + str(batch_idx) + '_ori.jpg'
            # cv2.imwrite(name0, tmpss0)
            #
            # superimposed_img = cv2.addWeighted(tmpss0, 0.6, heatmap, 0.4, 0)

        newBatch = np.array(newBatch).transpose(0, 3, 1, 2)
        # print (newBatch.shape)
        return self.to_var(torch.from_numpy(newBatch).float())

    def train(self):
        self.load_model(self.config['ourmodel_dir'])
        test_target_test = self.test(self.target_loader_test, 'ourmodel_tsne')
        print(test_target_test)

        self.load_model(self.config['basemodel_dir'])
        test_target_test = self.test(self.target_loader_test, 'basemodel_tsne')
        print(test_target_test)

        self.load_model(self.config['multitask_dir'])
        test_target_test = self.test(self.target_loader_test, 'multitask_tsne')
        print(test_target_test)

    def test(self, dataset_loader, save_name):
        self.classfier.eval()
        self.featureExactor.eval()
        self.featureExactor1.eval()
        test_loss = 0
        correct = 0
        num_label = np.zeros((200, 1))
        right_label = np.zeros((200, 1))
        for data, target in dataset_loader:
            data, target = data.cuda(), target.cuda()

            data, target = Variable(data, volatile=True), Variable(target)

            tmp_data, feature1 = self.featureExactor(data)
            out, _ = self.classfier(tmp_data)

            if self.config['spatial']:
                try:
                    # print('spatial')
                    source_data0 = self.cropBatch(data, feature1)
                    # crop
                    out1_re0, _ = self.featureExactor1(source_data0)

                    out10, _ = self.classfier(out1_re0)

                    out = self.config['ori'] * out + (1 - self.config['ori']) * out10
                except IndexError:
                    out = out

            # sum up batch loss
            test_loss += torch.nn.functional.cross_entropy(out, target, size_average=False).data[0]

            # get the index of the max log-probability
            pred = out.data.max(1, keepdim=True)[1]

            is_right = pred.eq(target.data.view_as(pred)).cpu()

            correct += is_right.sum()

            target = target.data.cpu()
            for i in range(len(target)):
                num_label[target[i]] = num_label[target[i]] + 1
                if (is_right[i]).numpy():
                    right_label[target[i]] = right_label[target[i]] + 1

        # print(num_label)
        # print(right_label)

        test_loss /= len(dataset_loader.dataset)

        # print(correct)
        # print(len(dataset_loader.dataset))
        import pickle
        with open(self.config['save_dir'] + save_name + 'right', 'wb') as fp:
            pickle.dump(right_label, fp)
        with open(self.config['save_dir'] + 'sum', 'wb') as fp:
            pickle.dump(num_label, fp)
        return {
            'average_loss': test_loss,
            'correct': correct,
            'total': len(dataset_loader.dataset),
            'accuracy': 100. * correct / len(dataset_loader.dataset)
        }
