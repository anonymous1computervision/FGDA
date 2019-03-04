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

        # torch.cuda.set_device(config['cuda'])

        self.config = config

        self.CUDA = True
        self.BATCH_SIZE = config['batch_size']
        self.EPOCHS = config['epoch']
        self.is_semi = False

        self.source_loader = get_dataloader(batch_size=self.BATCH_SIZE[0], domain=config['source'])
        print (len(self.source_loader))
        self.target_loader = get_dataloader_target(batch_size=self.BATCH_SIZE[1], domain=config['target'],
                                                   istrain='train')
        print (len(self.target_loader))

        self.source_loader_test = self.source_loader
        print (len(self.source_loader_test))
        self.target_loader_test = get_dataloader_target(batch_size=self.BATCH_SIZE[1], domain=config['target'],
                                                        istrain='test')
        print (len(self.target_loader_test))

        self.LEARNING_RATE = config['lr']
        self.WEIGHT_DECAY = config['weight_decay']
        self.MOMENTUM = config['momentum']

        self.build_model()
        self.load_model()

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

        self.concat = concatNet(200)
        self.concat.cuda()

        # support different learning rate according to CORAL paper
        # i.e. 10 times learning rate for the last two fc layers.
        self.optimizer_F = torch.optim.SGD(self.featureExactor.parameters(),
                                           lr=self.config['lr_f'] * self.LEARNING_RATE,
                                           momentum=self.MOMENTUM)
        self.optimizer_F1 = torch.optim.SGD(self.featureExactor1.parameters(),
                                            lr=self.config['lr_f'] * self.LEARNING_RATE,
                                            momentum=self.MOMENTUM)
        self.optimizer_C = torch.optim.SGD(self.classfier.parameters(), lr=self.config['lr_c'] * self.LEARNING_RATE,
                                           momentum=self.MOMENTUM)
        self.optimizer_D0 = torch.optim.SGD(self.domain[0].parameters(), lr=self.config['lr_d'] * self.LEARNING_RATE,
                                            momentum=self.MOMENTUM)
        self.optimizer_D1 = torch.optim.SGD(self.domain[1].parameters(), lr=self.config['lr_d'] * self.LEARNING_RATE,
                                            momentum=self.MOMENTUM)
        self.optimizer_cat = torch.optim.SGD(self.concat.parameters(), lr=self.config['lr_c'] * self.LEARNING_RATE,
                                             momentum=self.MOMENTUM)

        self.optimizer_main = torch.optim.SGD(
            list(self.featureExactor.parameters()) + list(self.classfier.parameters()),
            lr=self.config['lr_f'] * self.LEARNING_RATE,
            momentum=self.MOMENTUM,
            weight_decay=self.WEIGHT_DECAY)

        self.l1 = nn.L1Loss(size_average=True)
        self.cos = classInvariant()

    # load AlexNet pre-trained model
    def load_pretrained(self):
        url = '/unsullied/sharefs/wangyimu/data/pretrained_model/alexnet/alexnet-owt-4df8aa71.pth'
        pretrained_dict = torch.load(url)
        # model_dict = self.featureExactor.module.state_dict()
        model_dict = self.featureExactor.state_dict()

        # filter out unmatch dict and delete last fc bias, weight
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.featureExactor.load_state_dict(model_dict)
        # self.featureExactor.module.load_state_dict(model_dict)

        pretrained_dict = torch.load(url)
        # model_dict = self.featureExactor.module.state_dict()
        model_dict = self.featureExactor1.state_dict()

        # filter out unmatch dict and delete last fc bias, weight
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.featureExactor1.load_state_dict(model_dict)
        # self.featureExactor.module.load_state_dict(model_dict)

        pretrained_dict = torch.load(url)
        # model_dict = self.featureExactor.module.state_dict()
        model_dict = self.classfier.state_dict()

        # filter out unmatch dict and delete last fc bias, weight
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.classfier.load_state_dict(model_dict)
        # self.featureExactor.module.load_state_dict(model_dict)

        # modeldir = '/unsullied/sharefs/wangyimu/data/results/fine-grained/semi-supervised/ourdataset/sku_shelf/best/44.3243/'
        # self.featureExactor.load_state_dict(torch.load(modeldir + '/featureExactor_checkpoint.tar'))
        # self.featureExactor1.load_state_dict(torch.load(modeldir + '/featureExactor1_checkpoint.tar'))
        # self.classfier.load_state_dict(torch.load(modeldir + '/classifier_checkpoint.tar'))

    def load_model(self):
        self.load_pretrained()
        # if args.load is not None:
        #     utils.load_net(self.featureExactor.module, args.load)
        # else:
        #     load_pretrained(self.featureExactor.module)

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def grad_reset(self):
        self.optimizer_C.zero_grad()
        self.optimizer_F.zero_grad()
        self.optimizer_F1.zero_grad()
        self.optimizer_D0.zero_grad()
        self.optimizer_D1.zero_grad()
        self.optimizer_cat.zero_grad()
        self.optimizer_main.zero_grad()

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

    def train_perEpoch(self, epoch):
        result = []

        # Expected size : xs -> (batch_size, 3, 300, 300), ys -> (batch_size)
        source, target = list(enumerate(self.source_loader)), list(enumerate(self.target_loader))
        target_test = list(enumerate(self.target_loader_test))

        # train_steps = min(len(source), len(target))
        train_steps = len(source)
        # print (len(source))
        # print (len(target))

        for batch_idx in range(train_steps):
            _, (source_data, source_label) = source[batch_idx]
            _, (target_data, target_label) = target[batch_idx % len(target)]
            _, (target_data_test, target_label_test) = target_test[batch_idx % len(target_test)]

            # for target_train in source_label:
            #     if target_train in target_label_test:
            #         print('error, dataset overlapped')
            # for target_train in target_label:
            #     if target_train in target_label_test:
            #         print('error, dataset overlapped')

            # print(target_label)
            # print(target_label_test)

            source_data = self.to_var(source_data)
            source_label = self.to_var(source_label)
            target_data = self.to_var(target_data)
            target_label = self.to_var(target_label)

            target_data_test = self.to_var(target_data_test)
            target_label_test = self.to_var(target_label_test)

            if self.config['domain']:
                if batch_idx % self.config['FC/D'] == 0:
                    # train D
                    self.grad_reset()

                    _, feature1 = self.featureExactor(source_data)
                    _, feature2 = self.featureExactor(target_data)

                    if self.config['domainloss'] == 1:
                        domain1 = self.domain[0](feature1)
                        domain2 = self.domain[1](feature2)
                        # domain2 = self.domain[0](feature2)
                        domain3 = self.domain[1](feature3)
                        tmp_a = self.to_var(torch.FloatTensor(np.ones(domain1.shape)))
                        tmp_b = self.to_var(torch.FloatTensor(np.ones(domain2.shape)))
                        tmp_c = self.to_var(torch.FloatTensor(np.ones(domain3.shape)))
                        domaindisloss0 = self.l1(domain1, tmp_a)
                        domaindisloss1 = self.l1(domain2, tmp_b) + self.l1(domain3, tmp_c)
                    else:
                        domain11 = self.domain[0](feature1)
                        domain12 = self.domain[1](feature1)
                        domain21 = self.domain[0](feature2)
                        domain22 = self.domain[1](feature2)

                        tmp_1a = self.to_var(torch.FloatTensor(np.ones(domain11.shape)))
                        tmp_1b = self.to_var(torch.FloatTensor(np.zeros(domain11.shape)))
                        tmp_2a = self.to_var(torch.FloatTensor(np.zeros(domain21.shape)))
                        tmp_2b = self.to_var(torch.FloatTensor(np.ones(domain21.shape)))

                        domaindisloss0 = self.l1(domain11, tmp_1a) + self.l1(domain12, tmp_1b)
                        domaindisloss1 = self.l1(domain21, tmp_2a) + self.l1(domain22, tmp_2b)
                    # domaindisloss0.backward()
                    # domaindisloss1.backward()
                    domaindisloss_d = domaindisloss0 + domaindisloss1
                    domaindisloss_d.backward()

                    self.optimizer_D0.step()
                    self.optimizer_D1.step()
            else:
                domaindisloss_d = 0

            # train F
            self.grad_reset()

            out1_re, feature1 = self.featureExactor(source_data)
            out2_re, feature2 = self.featureExactor(target_data)

            out1, mmd_1 = self.classfier(out1_re)
            out2, mmd_2 = self.classfier(out2_re)

            if self.config['spatial']:
                try:
                    # print('spatial')
                    source_data_spatial = self.cropBatch(source_data, feature1)
                    target_data_spatial = self.cropBatch(target_data, feature2)

                    # crop
                    out1_re_spatial, feature1_spatial = self.featureExactor1(source_data_spatial)
                    out2_re_spatial, feature2_spatial = self.featureExactor1(target_data_spatial)

                    out1_spatial, _ = self.classfier(out1_re_spatial)
                    out2_spatial, _ = self.classfier(out2_re_spatial)

                    if self.config['concat']:
                        out1 = self.concat(out1, out1_spatial)
                        out2 = self.concat(out2, out2_spatial)
                    else:
                        out1 = self.config['ori'] * out1 + (1 - self.config['ori']) * out1_spatial
                        out2 = self.config['ori'] * out2 + (1 - self.config['ori']) * out2_spatial
                except IndexError:
                    out1 = out1
                    out2 = out2

            classification_loss = torch.nn.functional.cross_entropy(out1, source_label) \
                                  + torch.nn.functional.cross_entropy(out2, target_label)
            class_invariant = self.cos(mmd_1, mmd_2, source_label.data, target_label.data)

            if self.config['coral']:
                coral_loss = CORAL(out1, out2)
            else:
                coral_loss = 0
            if self.config['mmd']:
                if self.config['batch_size'][0] > self.config['batch_size'][1]:
                    mmd_loss = torch.abs(mmd(mmd_2, mmd_1))
                    # mmd_loss = mmd(mmd_2, mmd_1)
                else:
                    mmd_loss = torch.abs(mmd(mmd_1, mmd_2))
                mmd_loss = mmd_loss * mmd_loss
            else:
                mmd_loss = 0

            if self.config['domain']:
                if self.config['fadomainloss'] == 1:
                    domain11 = self.domain[0](feature1)
                    domain12 = self.domain[1](feature1)
                    domain21 = self.domain[0](feature2)
                    domain22 = self.domain[1](feature2)
                    loss1 = torch.abs(domain11 - domain12)
                    loss1 = loss1.sum() / self.BATCH_SIZE[0]
                    loss2 = torch.abs(domain21 - domain22)
                    loss2 = loss2.sum() / self.BATCH_SIZE[1]
                else:
                    domain11 = self.domain[0](feature1)
                    domain12 = self.domain[1](feature1)
                    domain21 = self.domain[0](feature2)
                    domain22 = self.domain[1](feature2)
                    tmp_a1 = self.to_var(torch.FloatTensor(np.zeros(domain11.shape)))
                    tmp_a2 = self.to_var(torch.FloatTensor(np.ones(domain11.shape)))
                    tmp_b1 = self.to_var(torch.FloatTensor(np.zeros(domain21.shape)))
                    tmp_b2 = self.to_var(torch.FloatTensor(np.ones(domain21.shape)))
                    loss1 = torch.abs(domain11 - tmp_a1) + torch.abs(domain12 - tmp_a2)
                    loss1 = loss1.sum() / self.BATCH_SIZE[0]
                    loss2 = torch.abs(domain21 - tmp_b1) + torch.abs(domain22 - tmp_b2)
                    loss2 = loss2.sum() / self.BATCH_SIZE[1]

                if self.config['spatial_dis']:
                    domain11_spatial = self.domain[0](feature1_spatial)
                    domain12_spatial = self.domain[1](feature1_spatial)
                    domain21_spatial = self.domain[0](feature2_spatial)
                    domain22_spatial = self.domain[1](feature2_spatial)
                    loss1_spatial = torch.abs(domain11_spatial - domain12_spatial)
                    loss1_spatial = loss1_spatial.sum() / self.BATCH_SIZE[0]
                    loss2_spatial = torch.abs(domain21_spatial - domain22_spatial)
                    loss2_spatial = loss2_spatial.sum() / self.BATCH_SIZE[1]
                    domaindisloss = loss1 + loss2 + loss1_spatial + loss2_spatial
                    domaindisloss /= 4
                else:
                    domaindisloss = loss1 + loss2
                    domaindisloss /= 2
            else:
                domaindisloss = 0

            # f_loss = self._lambda * coral_loss + classification_loss + 0 * mmd_loss
            # f_loss = 0 * coral_loss + classification_loss + 0 * mmd_loss
            f_loss = self.config['coral'] * coral_loss + classification_loss + self.config['mmd'] * mmd_loss + \
                     self.config['domain'] * domaindisloss + self.config['class'] * class_invariant
            # f_loss = classification_loss + self.config['domain'] * domaindisloss
            # f_loss.backward(retain_graph=True)
            f_loss.backward()

            # self.optimizer_main.step()

            self.optimizer_F.step()
            if self.config['spatial']:
                self.optimizer_F1.step()
                
            c_loss = classification_loss
            # c_loss.backward()

            self.optimizer_C.step()
            if self.config['concat']:
                self.optimizer_cat.step()

            if self.config['domain']:
                result.append({
                    'epoch': epoch,
                    'step': batch_idx + 1,
                    'total_steps': train_steps,
                    # 'lambda': self._lambda,
                    # 'coral_loss': coral_loss.data[0],
                    'classification_loss': classification_loss.data[0],
                    # 'mmd_loss': mmd_loss.data[0],
                    'domain_loss': domaindisloss.data[0],
                    # 'domain_loss': domaindisloss,
                    'class_loss': class_invariant.data[0],
                    'f_loss': f_loss.data[0],
                    'c_loss': c_loss.data[0],
                    'd_loss': domaindisloss_d.data[0]
                    # 'd_loss': domaindisloss_d
                })
            else:
                result.append({
                    'epoch': epoch,
                    'step': batch_idx + 1,
                    'total_steps': train_steps,
                    # 'lambda': self._lambda,
                    # 'coral_loss': coral_loss.data[0],
                    'classification_loss': classification_loss.data[0],
                    # 'mmd_loss': mmd_loss.data[0],
                    # 'domain_loss': domaindisloss.data[0],
                    'class_loss': class_invariant.data[0],
                    'domain_loss': domaindisloss,
                    'f_loss': f_loss.data[0],
                    'c_loss': c_loss.data[0],
                    # 'd_loss': domaindisloss_d.data[0]
                    'd_loss': domaindisloss_d
                })

            # tqdm_result = 'Train Epoch: {:2d} [{:2d}/{:2d}]\tLambda: {:.4f}, Class: {:.6f}, CORAL: {:.6f}, Total_Loss: {:.6f}\nmmd_loss : {:.6f}'.format(
            #     epoch,
            #     batch_idx + 1,
            #     train_steps,
            #     _lambda,
            #     classification_loss.data[0],
            #     coral_loss.data[0],
            #     sum_loss.data[0],
            #     mmd_loss.data[0]
            # )
            # tqdm.write(tqdm_result)

        return result

    def train(self):
        # print ('a ' + str(self.config['domain']))
        # print ('domainloss ' + str(self.config['domainloss']))
        # print ('fadomainloss ' + str(self.config['fadomainloss']))
        # print ('spatial ' + str(self.config['ori']))

        for key in self.config.keys():
            print (key + '\t' + str(self.config[key]))

        # self.featureExactor = torch.nn.DataParallel(self.featureExactor)
        # self.featureExactor1 = torch.nn.DataParallel(self.featureExactor1)
        # self.classfier = torch.nn.DataParallel(self.classfier)
        # self.domain[0] = torch.nn.DataParallel(self.domain[0])
        # self.domain[1] = torch.nn.DataParallel(self.domain[1])

        training_statistic = []
        testing_s_statistic = []
        testing_t_statistic = []

        max_acc = 0
        i = 0

        for e in tqdm(range(0, self.EPOCHS)):
            self._lambda = (e + 1) / self.EPOCHS
            # _lambda = 0.0
            res = self.train_perEpoch(e + 1)
            tqdm_result = '###EPOCH {}: Class: {:.6f}, domain: {:.6f}, cls-inv: {:.6f}, f_Loss: {:.6f}, c_Loss: {:.6f}, d_Loss: {:.6f}'.format(
                e + 1,
                sum(row['classification_loss'] / row['total_steps'] for row in res),
                # sum(row['coral_loss'] / row['total_steps'] for row in res),
                # sum(row['mmd_loss'] / row['total_steps'] for row in res),
                sum(row['domain_loss'] / row['total_steps'] for row in res),
                sum(row['class_loss'] / row['total_steps'] for row in res),
                sum(row['f_loss'] / row['total_steps'] for row in res),
                sum(row['c_loss'] / row['total_steps'] for row in res),
                sum(row['d_loss'] / row['total_steps'] for row in res),
            )
            tqdm.write(tqdm_result)

            training_statistic.append(res)

            test_source = self.test(self.source_loader_test, e)
            test_target = self.test(self.target_loader, e)
            test_target_test = self.test(self.target_loader_test, e)
            testing_s_statistic.append(test_source)
            testing_t_statistic.append(test_target)

            tqdm_result = '###Test Source: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                e + 1,
                test_source['average_loss'],
                test_source['correct'],
                test_source['total'],
                test_source['accuracy'],
            )
            tqdm.write(tqdm_result)
            tqdm_result = '###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                e + 1,
                test_target['average_loss'],
                test_target['correct'],
                test_target['total'],
                test_target['accuracy'],
            )
            tqdm.write(tqdm_result)
            tqdm_result = '###Test Target test: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                e + 1,
                test_target_test['average_loss'],
                test_target_test['correct'],
                test_target_test['total'],
                test_target_test['accuracy'],
            )
            tqdm.write(tqdm_result)

            if test_target_test['accuracy'] > max_acc:
                max_acc = test_target_test['accuracy']
                i = e

                root = '/unsullied/sharefs/wangyimu/data/results/fine-grained/semi-supervised/ourdataset/' + \
                       self.config[
                           'source'] + '_' + self.config['target']
                if not os.path.exists(root):
                    os.makedirs(root)
                root = root + '/best/' + '{:.4f}'.format(max_acc) + '/'
                if not os.path.exists(root):
                    os.makedirs(root)

                tqdm.write(utils.save_net(self.classfier, root + '/classifier_checkpoint.tar'))
                tqdm.write(utils.save_net(self.featureExactor, root + '/featureExactor_checkpoint.tar'))
                tqdm.write(utils.save_net(self.featureExactor1, root + '/featureExactor1_checkpoint.tar'))
                tqdm.write(utils.save_net(self.domain[0], root + '/domainDiscriminator0_checkpoint.tar'))
                tqdm.write(utils.save_net(self.domain[1], root + '/domainDiscriminator1_checkpoint.tar'))

            bestnow = '###Epoch: {},Accuracy: {:.2f}'.format(
                i,
                max_acc,
            )
            tqdm.write(bestnow)

        for key in self.config.keys():
            print (key + '\t' + str(self.config[key]))

        print (bestnow)

        tqdm.write(utils.save_net(self.classfier, root + '/classifier_final.tar'))
        tqdm.write(utils.save_net(self.featureExactor, root + '/featureExactor_final.tar'))
        tqdm.write(utils.save_net(self.featureExactor1, root + '/featureExactor1_final.tar'))
        tqdm.write(utils.save_net(self.domain[0], root + '/domainDiscriminator0_final.tar'))
        tqdm.write(utils.save_net(self.domain[1], root + '/domainDiscriminator1_final.tar'))
        print(utils.save(training_statistic, root + '/training_statistic.pkl'))
        print(utils.save(testing_s_statistic, root + '/testing_s_statistic.pkl'))
        print(utils.save(testing_t_statistic, root + '/testing_t_statistic.pkl'))
        # test_target_test = self.test(self.target_loader_test, 1)
        # print(test_target_test)

    def test(self, dataset_loader, e):
        self.classfier.eval()
        self.featureExactor.eval()
        self.concat.eval()
        test_loss = 0
        correct = 0
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

                    if self.config['concat']:
                        out = self.concat(out, out10)
                    else:
                        out = self.config['ori'] * out + (1 - self.config['ori']) * out10
                except IndexError:
                    out = out

            # print (out)
            # print (target)
            # sum up batch loss
            test_loss += torch.nn.functional.cross_entropy(out, target, size_average=False).data[0]

            # get the index of the max log-probability
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataset_loader.dataset)
	
	import scipy.io as sio
	

        # print(correct)
        # print(len(dataset_loader.dataset))
        return {
            'epoch': e,
            'average_loss': test_loss,
            'correct': correct,
            'total': len(dataset_loader.dataset),
            'accuracy': 100. * correct / len(dataset_loader.dataset)
     }
