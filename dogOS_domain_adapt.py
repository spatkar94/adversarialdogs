# Contributor: Sushant Patkar
# Email: patkar.sushant@nih.gov
# Sept 30, 2022
#
# By downloading or otherwise receiving the SOFTWARE, RECIPIENT may 
# use and/or redistribute the SOFTWARE, with or without modification, 
# subject to RECIPIENT’s agreement to the following terms:
# 
# 1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS 
# OF CANINE OR HUMAN SUBJECTS.  RECIPIENT is responsible for 
# compliance with all laws and regulations applicable to the use 
# of the SOFTWARE.
# 
# 2. The SOFTWARE that is distributed pursuant to this Agreement 
# has been created by United States Government employees. In 
# accordance with Title 17 of the United States Code, section 105, 
# the SOFTWARE is not subject to copyright protection in the 
# United States.  Other than copyright, all rights, title and 
# interest in the SOFTWARE shall remain with the PROVIDER.   
# 
# 3.	RECIPIENT agrees to acknowledge PROVIDER’s contribution and 
# the name of the author of the SOFTWARE in all written publications 
# containing any data or information regarding or resulting from use 
# of the SOFTWARE. 
# 
# 4.	THE SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT 
# ARE DISCLAIMED. IN NO EVENT SHALL THE PROVIDER OR THE INDIVIDUAL DEVELOPERS 
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
# THE POSSIBILITY OF SUCH DAMAGE.  
# 
# 5.	RECIPIENT agrees not to use any trademarks, service marks, trade names, 
# logos or product names of NCI or NIH to endorse or promote products derived 
# from the SOFTWARE without specific, prior and written permission.
# 
# 6.	For sake of clarity, and not by way of limitation, RECIPIENT may add its 
# own copyright statement to its modifications or derivative works of the SOFTWARE 
# and may provide additional or different license terms and conditions in its 
# sublicenses of modifications or derivative works of the SOFTWARE provided that 
# RECIPIENT’s use, reproduction, and distribution of the SOFTWARE otherwise complies 
# with the conditions stated in this Agreement. Whenever Recipient distributes or 
# redistributes the SOFTWARE, a copy of this Agreement must be included with 
# each copy of the SOFTWARE.

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from PIL import Image
import pickle
import re
import operator
import pandas as pd
import random
from itertools import cycle
import argparse


#define dataset class for reading in training validation and test whole slide image patches from dog OS tumors
class DogDataset(Dataset):
     def __init__(self, rootdir, level, transform = None, mode = None):
          self.rootdir = rootdir
          self.transform = transform
          self.slides = os.listdir(rootdir)
          files = [self.slides[i] + '/' + x for i in range(len(self.slides)) for x in os.listdir(os.path.join(rootdir,self.slides[i]))]
          file_names = [x for i in range(len(self.slides)) for x in os.listdir(os.path.join(rootdir,self.slides[i]))]
          nonvr_count = 0
           
          self.labels = []
          self.patches = []
          self.fnames = []
          self.slidenames = []

          #assigning class labels to each whole slide image patch based on overlapping pathologist annotations
          for i in range(len(files)):
               x = files[i]
               ws = float(re.findall(r"ws-\d.\d\d", x)[0].split("-")[1])
               tumor = re.findall(r"TumorOB-\d.\d\d", x)
               HN = re.findall(r"HN-\d.\d\d", x)
               H = re.findall(r"H-\d.\d\d", x)
               CB = re.findall(r"CB-\d.\d\d", x)
               FB = re.findall(r"FB-\d.\d\d", x)
               GC = re.findall(r"GC-\d.\d\d", x)
               VR = re.findall(r"VR-\d.\d\d", x)
               NA = re.findall(r"NA-\d.\d\d", x)
               other = re.findall(r"other-\d.\d\d",x)
               exclude = re.findall(r"exclude-\d.\d\d",x)

               areas = {'tumor':0.0,'H':0.0,'HN':0.0,'CB':0.0,'FB':0.0,'GC':0.0,'VR':0.0,'NA':0.0,'other':0.0}
               if len(tumor)>0:
                    areas['tumor'] = float(tumor[0].split('-')[1])
               if len(HN)>0:
                    areas['HN'] = float(HN[0].split('-')[1])
               if len(H)>0:
                    areas['H'] = float(H[0].split('-')[1])
               if len(CB)>0:
                    areas['CB'] = float(CB[0].split('-')[1])
               if len(FB)>0:
                    areas['FB'] = float(FB[0].split('-')[1])
               if len(GC)>0:
                    areas['GC'] = float(GC[0].split('-')[1])
               if len(VR)>0:
                    areas['VR'] = float(VR[0].split('-')[1])
                    if x.split('_')[0] == '2021-04-24 09.41.08/2021-04-24 09.41.08':
                         print(x.split('_')[0] + ' VR excluded')
                         nonvr_count += 1*(float(VR[0].split('-')[1]) > 0.15)
                         areas['VR'] = 0.0
               if len(NA)>0:
                    areas['NA'] = float(NA[0].split('-')[1])
               if len(other)>0:
                    areas['other'] = float(other[0].split('-')[1])
               if len(exclude)>0:
                    areas['exclude'] = float(exclude[0].split('-')[1])

               if areas['tumor'] > 0.0 and sum([areas[x] for x in ['H','HN','CB','FB','GC','VR','NA']]) > 0.15:
                    areas['tumor'] = 0.0

               lab = max(areas.items(), key=operator.itemgetter(1))[0]
               if lab == 'H':
                    lab = 'other'
               if lab == 'exclude':
                    lab = 'other'
               maxarea = areas[lab]               
               if maxarea > 0.15 and lab not in ['NA','exclude']:
                    self.patches.append(x)
                    self.labels.append(lab)
                    self.fnames.append(file_names[i])


          
          ll = list(range(len(self.labels)))
  

          
          #define train validation and test splits (80% train, 10% validation, 10% test)
          np.random.seed(123)
          testidx = np.random.choice(np.array(ll), int(len(ll)*0.1))
          xx = list(np.setdiff1d(np.array(ll), np.array(testidx)))
          validx = list(np.random.choice(xx,int(len(ll)*0.1)))
          trainidx = list(np.setdiff1d(np.array(xx), np.array(validx)))


          if mode == 'train':
               self.labels = [self.labels[i] for i in trainidx]
               self.fnames = [self.fnames[i] for i in trainidx]
               self.patches = [self.patches[i] for i in trainidx]
               
          elif mode == 'val':
               self.labels = [self.labels[i] for i in validx]
               self.fnames = [self.fnames[i] for i in validx]
               self.patches = [self.patches[i] for i in validx]
               
          elif mode == 'test':
               self.labels = [self.labels[i] for i in testidx]
               self.fnames = [self.fnames[i] for i in testidx]
               self.patches = [self.patches[i] for i in testidx]
               


          self.slidenames = [x.split('_')[0] for x in self.patches]
          self.slides = np.unique(self.slidenames)
          
              
          if level == "m0":
               pass

          self.classes = np.unique(self.labels)
          for i in range(len(self.labels)):
               self.labels[i] = int(np.where(self.classes == self.labels[i])[0])   

          print(nonvr_count)

     def __len__(self):
          return(len(self.labels))

     def __getitem__(self,idx):
          img = Image.open(os.path.join(self.rootdir,self.patches[idx]))
          if self.transform:
               img = self.transform(img)
          return img, self.labels[idx]

#define dataset class for reading in training and test whole slide image patches from human OS tumors
class HumanDataset(Dataset):
     def __init__(self, rootdir, level, transform = None, mode = None):
          self.rootdir = rootdir
          self.transform = transform
          self.slides = os.listdir(rootdir)
          files = [self.slides[i] + '/' + x for i in range(len(self.slides)) for x in os.listdir(os.path.join(rootdir,self.slides[i]))]
          file_names = [x for i in range(len(self.slides)) for x in os.listdir(os.path.join(rootdir,self.slides[i]))]

           
          self.labels = []
          self.patches = []
          self.fnames = []
          self.slidenames = []
          for i in range(len(files)):
               x = files[i]
               ws = float(re.findall(r"ws-\d.\d\d", x)[0].split("-")[1])
               tumor = re.findall(r"TumorOB-\d.\d\d", x)
               HN = re.findall(r"HN-\d.\d\d", x)
               H = re.findall(r"H-\d.\d\d", x)
               CB = re.findall(r"CB-\d.\d\d", x)
               FB = re.findall(r"FB-\d.\d\d", x)
               GC = re.findall(r"GC-\d.\d\d", x)
               VR = re.findall(r"VR-\d.\d\d", x)
               NA = re.findall(r"NA-\d.\d\d", x)
               other = re.findall(r"other-\d.\d\d",x)
               exclude = re.findall(r"exclude-\d.\d\d",x)

               areas = {'tumor':0.0,'H':0.0,'HN':0.0,'CB':0.0,'FB':0.0,'GC':0.0,'VR':0.0,'NA':0.0,'other':0.0}
               if len(tumor)>0:
                    areas['tumor'] = float(tumor[0].split('-')[1])
               if len(HN)>0:
                    areas['HN'] = float(HN[0].split('-')[1])
               if len(H)>0:
                    areas['H'] = float(H[0].split('-')[1])
               if len(CB)>0:
                    areas['CB'] = float(CB[0].split('-')[1])
               if len(FB)>0:
                    areas['FB'] = float(FB[0].split('-')[1])
               if len(GC)>0:
                    areas['GC'] = float(GC[0].split('-')[1])
               if len(VR)>0:
                    areas['VR'] = float(VR[0].split('-')[1])
               if len(NA)>0:
                    areas['NA'] = float(NA[0].split('-')[1])
               if len(other)>0:
                    areas['other'] = float(other[0].split('-')[1])
               if len(exclude)>0:
                    areas['exclude'] = float(exclude[0].split('-')[1])

               if areas['tumor'] > 0.0 and sum([areas[x] for x in ['H','HN','CB','FB','GC','VR','NA']]) > 0.15:
                    areas['tumor'] = 0.0

               lab = max(areas.items(), key=operator.itemgetter(1))[0]
               if lab == 'H':
                    lab = 'other'
               if lab == 'exclude':
                    lab = 'other'
               maxarea = areas[lab]               
               if maxarea > 0.15 and lab not in ['NA','exclude']:
                    self.patches.append(x)
                    self.labels.append(lab)
                    self.fnames.append(file_names[i])


          
          #ll = list(np.where(np.logical_or(np.array(self.labels) != 'NA', np.array(self.labels) != 'exclude'))[0])
          ll = list(range(len(self.labels)))
          
          self.slidenames = [x.split('_')[0] for x in self.patches]
          self.slides = np.unique(self.slidenames)



          #define train and test splits (~3% train (2000 patches), ~97% test)
          np.random.seed(123)
          trainidx = np.random.choice(np.array(ll), 2000)
          testidx = list(np.setdiff1d(np.array(ll), np.array(trainidx)))

          
          if mode == 'train':
               self.labels = [self.labels[i] for i in trainidx]
               self.fnames = [self.fnames[i] for i in trainidx]
               self.patches = [self.patches[i] for i in trainidx]

          elif mode == 'unlabeled':
               unlabeledidx = np.random.choice(testidx, int(len(testidx)*0.5))
               self.labels = [self.labels[i] for i in unlabeledidx]
               self.fnames = [self.fnames[i] for i in unlabeledidx]
               self.patches = [self.patches[i] for i in unlabeledidx]
               
               
          elif mode == 'test':
               self.labels = [self.labels[i] for i in testidx]
               self.fnames = [self.fnames[i] for i in testidx]
               self.patches = [self.patches[i] for i in testidx]
               


          self.slidenames = [x.split('_')[0] for x in self.patches]
          self.slides = np.unique(self.slidenames)
          
              
          if level == "m0":
               pass

          self.classes = np.unique(self.labels)
          for i in range(len(self.labels)):
               self.labels[i] = int(np.where(self.classes == self.labels[i])[0])    

     def __len__(self):
          return(len(self.labels))

     def __getitem__(self,idx):
          img = Image.open(os.path.join(self.rootdir,self.patches[idx]))
          if self.transform:
               img = self.transform(img)
          return img, self.labels[idx]

#define gradient reversal layer for domain classifier
class GradientReversal(torch.autograd.Function):
     @staticmethod
     def forward(ctx, x):
          return x.view_as(x)

     @staticmethod
     def backward(ctx, grad_output):
          return grad_output.neg()*GradientReversal.lambd, None 
          
def grad_reverse(x, lambd):
     GradientReversal.lambd = lambd
     return GradientReversal.apply(x)


#define feature extraction backbone
class FeatureExtractor(torch.nn.Module):
     def __init__(self, backbone, use_pretrained=True, freeze_weights = False, freeze_point = 8):
          super(FeatureExtractor, self).__init__()
          self.backbone = None
          if backbone == "resnet18":
               self.backbone = models.resnet18(pretrained=use_pretrained)
          if backbone == "resnet34":
               self.backbone = models.resnet34(pretrained=use_pretrained)
          if backbone == "resnet50":
               self.backbone = models.resnet50(pretrained=use_pretrained)
          if backbone == "resnet101":
               self.backbone = models.resnet101(pretrained=use_pretrained)
          if backbone == "resnet152":
               self.backbone = models.resnet152(pretrained=use_pretrained)

          if freeze_weights:
               ct = 0
               for child in self.backbone.children():
                    ct += 1
                    if ct < freeze_point:
                         for param in child.parameters():
                              param.requires_grad = False

          num_ftr = self.backbone.fc.in_features

          self.backbone.fc = torch.nn.Identity()
          self.out_dim = num_ftr


     def forward(self, x):
          return self.backbone(x)

#define histological subtype classifier/domain classifier
class Classifier(torch.nn.Module):
     def __init__(self, indim, num_classes, hidden_layers = 1, hdims = [512], reverse_grad = False, lambda_ = 1.0):
          super(Classifier, self).__init__()
          self.reverse_grad = reverse_grad
          self.lambda_ = lambda_
          modules = []
          if hidden_layers > 0:
               inp = indim
               for i in range(hidden_layers):
                    modules.append(torch.nn.Linear(inp, hdims[i]))
                    modules.append(torch.nn.ReLU())
                    modules.append(torch.nn.Dropout(p=0.5))
                    inp = hdims[i]
               modules.append(torch.nn.Linear(inp, num_classes))
               
          else:
               modules.append(torch.nn.Linear(indim, num_classes))
               
          self.fc = torch.nn.Sequential(*modules)

     def set_lambda(self, lambda_):
          self.lambda_ = lambda_
          

     def forward(self, x):
          if self.reverse_grad:
               x = grad_reverse(x, self.lambda_)
          return self.fc(x)


#define sampling strategy for sampling minibatches during training
class SlideSampler(Sampler):
     def __init__(self, dataset):
          self.num_samples = len(dataset)
          self.labels = dataset.labels
          self.classes = np.unique(self.labels)
          self.indices = []
          
          slides = np.random.permutation(np.unique(dataset.slidenames))


          #sampling tiles by slide
          j = 0
          for i in tqdm(range(0,self.num_samples,32)):
               idx = list(np.random.choice(np.where(np.array(dataset.slidenames) == slides[j%len(slides)])[0],32))
               self.indices.append(idx)
               j = j + 1
          self.indices = [y for x in self.indices for y in x]  
       
       

     def __iter__(self):
          return iter(self.indices)


     def __len__(self):
          return len(self.indices)

#Unsupervised domain adatation (Ganin & Lempinsky et al 2014)
def train_unsupervised(feature_extractor, classifier, discriminator, loss_fn, optimizer, source_loader, target_loader):
     feature_extractor.train()
     classifier.train()
     discriminator.train()
     print(discriminator.lambda_)
     print(optimizer.lr)
     batches = zip(source_loader, cycle(target_loader)) 
     n_batches = max(len(source_loader), len(target_loader))
     pbar = tqdm(batches, total = n_batches)
     epoch_loss = 0.0
     edomain_loss = 0.0
     for (source_x, ys), (target_x, _) in pbar:
          x = torch.cat([source_x, target_x])
          x = x.cuda()
          domain_y = torch.cat([torch.zeros(source_x.shape[0]), torch.ones(target_x.shape[0])])
          #domain_y = domain_y.long()
          domain_y = domain_y.cuda()
          y = ys
          y = y.cuda()

          features = feature_extractor(x).view(x.shape[0], -1)
          domain_preds = discriminator(features).squeeze()
          label_preds = classifier(features[:(source_x.shape[0])])

          cls_loss = loss_fn(label_preds, y)
          domain_loss = torch.nn.functional.binary_cross_entropy_with_logits(domain_preds, domain_y)
          loss = domain_loss + cls_loss

          optimizer.zero_grad()
          loss.backward()
          epoch_loss += cls_loss.item()/n_batches
          edomain_loss += domain_loss.item()/n_batches
          optimizer.step()
          pbar.set_description('Train Epoch {}: classifier Loss@:{:.5f}, domain Loss@:{:.5f}'.format(epoch, cls_loss, domain_loss))

     print('Train Epoch {}: overall classifier Loss@:{:.5f} overall domain Loss@:{:.5f}'.format(epoch, epoch_loss, edomain_loss))
     return epoch_loss

#Semi-supervised domain adaptation
def train_semisupervised(feature_extractor, classifier, discriminator, loss_fn, optimizer, source_loader, target_loader, unlabeledloader):
     feature_extractor.train()
     classifier.train()
     discriminator.train()
     print(discriminator.lambda_)
     print(optimizer.lr)
     #batches = zip(source_loader, target_loader)
     batches = zip(source_loader, cycle(target_loader), cycle(unlabeledloader)) 
     n_batches = max(len(source_loader), len(target_loader), len(unlabeledloader))
     pbar = tqdm(batches, total = n_batches)
     epoch_loss = 0.0
     edomain_loss = 0.0
     for (source_x, ys), (target_x, yt), (target_x2, _) in pbar:
          x = torch.cat([source_x, target_x, target_x2])
          x = x.cuda()
          domain_y = torch.cat([torch.zeros(source_x.shape[0]), torch.ones(target_x.shape[0] + target_x2.shape[0])])
          #domain_y = domain_y.long()
          domain_y = domain_y.cuda()
          y = torch.cat([ys, yt])
          y = y.cuda()

          features = feature_extractor(x).view(x.shape[0], -1)
          domain_preds = discriminator(features).squeeze()
          label_preds = classifier(features[:(source_x.shape[0] + target_x.shape[0])])

          cls_loss = loss_fn(label_preds, y)
          domain_loss = torch.nn.functional.binary_cross_entropy_with_logits(domain_preds, domain_y)
          loss = domain_loss + cls_loss

          optimizer.zero_grad()
          loss.backward()
          epoch_loss += cls_loss.item()/n_batches
          edomain_loss += domain_loss.item()/n_batches
          optimizer.step()
          pbar.set_description('Train Epoch {}: classifier Loss@:{:.5f}, domain Loss@:{:.5f}'.format(epoch, cls_loss, domain_loss))

     print('Train Epoch {}: overall classifier Loss@:{:.5f} overall domain Loss@:{:.5f}'.format(epoch, epoch_loss, edomain_loss))
     return epoch_loss

#supervised domain adaptation
def train_supervised(feature_extractor, classifier, discriminator, loss_fn, optimizer, source_loader, target_loader):
     feature_extractor.train()
     classifier.train()
     discriminator.train()
     print(discriminator.lambda_)
     print(optimizer.lr)
     #batches = zip(source_loader, target_loader)
     batches = zip(source_loader, cycle(target_loader)) 
     n_batches = max(len(source_loader), len(target_loader))
     pbar = tqdm(batches, total = n_batches)
     epoch_loss = 0.0
     edomain_loss = 0.0
     elosses = []
     for (source_x, ys), (target_x, yt) in pbar:
          x = torch.cat([source_x, target_x])
          x = x.cuda()
          domain_y = torch.cat([torch.zeros(source_x.shape[0]), torch.ones(target_x.shape[0])])
          #domain_y = domain_y.long()
          domain_y = domain_y.cuda()
          y = torch.cat([ys, yt])
          y = y.cuda()

          features = feature_extractor(x).view(x.shape[0], -1)
          domain_preds = discriminator(features).squeeze()
          label_preds = classifier(features)

          cls_loss = loss_fn(label_preds, y)
          domain_loss = torch.nn.functional.binary_cross_entropy_with_logits(domain_preds, domain_y)
          loss = domain_loss + cls_loss

          optimizer.zero_grad()
          loss.backward()
          elosses.append(cls_loss.item())
          epoch_loss += cls_loss.item()/n_batches
          edomain_loss += domain_loss.item()/n_batches
          optimizer.step()
          pbar.set_description('Train Epoch {}: classifier Loss@:{:.5f}, domain Loss@:{:.5f}'.format(epoch, cls_loss, domain_loss))

     print('Train Epoch {}: overall classifier Loss@:{:.5f} overall domain Loss@:{:.5f}'.format(epoch, epoch_loss, edomain_loss))
     return epoch_loss

#evaluation of classifier predictions
def test(feature_extractor, classifier, test_data_loader, loss_fn, epoch, classes):
     feature_extractor.eval()
     classifier.eval()
     mPrec, mRecall, mean_f1 =  0.0, 0.0, 0.0
     epoch_loss = 0.0
     elosses = []
     preds = []
     targets = []
     per_class_prec = {}
     per_class_recall  = {}
     
     per_class_prec = {0:0.0,1:0.0,2:0.0,3:0.0,4:0.0,5:0.0,6:0.0}
     per_class_recall = {0:0.0,1:0.0,2:0.0,3:0.0,4:0.0,5:0.0,6:0.0}

     with torch.no_grad():
          test_bar = tqdm(test_data_loader)
          for data, y in test_bar:
               data, y = data.cuda(non_blocking=True), y.cuda(non_blocking=True)
               features = feature_extractor(data)
               label_preds = classifier(features)
               #domain_preds = discriminator(features)
               cls_loss = loss_fn(label_preds, y)
               loss = cls_loss
               elosses.append(cls_loss.item())
               epoch_loss += cls_loss.item()/len(test_data_loader)
               _, pred_labels = torch.max(label_preds, dim=1)
               preds.append(list(pred_labels.cpu().numpy()))
               targets.append(list(y.cpu().numpy()))
               
               test_bar.set_description('Test Epoch {} classifier Loss@:{:.5f}'.format(epoch, cls_loss))
          preds = np.array([y for x in preds for y in x] )
          targets = np.array([y for x in targets for y in x])

          assert len(preds) == len(targets)
          cm = confusion_matrix(targets,preds)
          values, counts = np.unique(targets, return_counts=True)
          print([classes[l] for l in values])
          print(counts)

          values, counts = np.unique(preds, return_counts=True)
          print([classes[l] for l in values])
          print(counts)
          print(cm)
          
          for c in np.unique(targets):
               n_pred = sum(np.array(preds) == c)*1.0
               if n_pred > 0:
                    true_pred = sum((np.array(preds) == np.array(targets))*(np.array(preds) == c))
                    per_class_prec[c] = true_pred/n_pred

          
          for c in np.unique(targets):
               n_target= sum(np.array(targets) == c)*1.0
               if n_target > 0:
                    true_pred = sum((np.array(preds) == np.array(targets))*(np.array(targets) == c))
                    per_class_recall[c] = true_pred/n_target



          per_class_f1 = 2*np.array(list(per_class_prec.values()))*np.array(list(per_class_recall.values()))/(np.array(list(per_class_prec.values())) + np.array(list(per_class_recall.values())))
          mPrec = np.nanmean(np.array(list(per_class_prec.values())))
          mRecall = np.nanmean(np.array(list(per_class_recall.values())))
          mF1 = np.nanmean(per_class_f1)
     print('Test Epoch {}: overall classifier loss@:{:.5f} mPrec@:{:.2f} mRecall@:{:.2f} mF1@:{:.2f}'.format(epoch, epoch_loss, mPrec, mRecall, mF1))
     print(per_class_prec)
     print(per_class_recall)
     print(per_class_f1)
     #record_keeper.update_records({"monitor_f1": mean_f1}, epoch, input_group_name_for_non_objects = "mean_f1")
     #record_keeper.save_records()
     return epoch_loss, mPrec, mRecall, mF1, list(per_class_prec.values()), list(per_class_recall.values()), list(per_class_f1), elosses


#function to save model weights
def save_model(model, name):
     model_folder = "saved_models"
     if not os.path.exists(model_folder):
          os.makedirs(model_folder)

     torch.save(model.state_dict(), "{}/{}_model_best.pth".format(model_folder,name))


#main function
if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--source", type=str, help="path to source data (dogs)")
     parser.add_argument("--target", type=str, help="path to target data (humans)")
     args = parser.parse_args()
     SOURCE_DATA = args.source
     TARGET_DATA = args.target
     level0 = "m0"
     base = FeatureExtractor(backbone = 'resnet50', use_pretrained = True, freeze_weights=False)
     os_classifier = Classifier(indim = base.out_dim, num_classes = 7, hidden_layers = 0)
     domain_classifier = Classifier(indim = base.out_dim, num_classes = 1, hidden_layers = 0, reverse_grad=True)

     print(os_classifier)
     print(domain_classifier)

     base = base.cuda()
     os_classifier = os_classifier.cuda()
     domain_classifier = domain_classifier.cuda()
     
     #normalize pixel intensities to follow standard normal range (mean 0, std ~ 1)
     normalize = transforms.Normalize(mean=[0.8938, 0.5708, 0.7944], std = [0.1163, 0.1528, 0.0885])

     #data augmentation transforms
     train_transforms = transforms.Compose([
          transforms.Resize((224,224)),
          #transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomVerticalFlip(p=0.5),
          #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
          #transforms.RandomGrayscale(p=0.2),
          transforms.ToTensor(),
          normalize
     ])

     basic_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])


     sourcedataset = DogDataset(SOURCE_DATA, level = level0, transform = train_transforms, mode = "train")
     targetdataset = HumanDataset(TARGET_DATA, level = level0, transform = train_transforms, mode = "train")

     valdataset = DogDataset(SOURCE_DATA, level = level0, transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize]), mode = "val")
     testdataset = DogDataset(SOURCE_DATA, level = level0, transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize]), mode = "test")
     
     testdataset2 = HumanDataset(TARGET_DATA, level = level0, transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize]), mode = "test")

     ss = SlideSampler(sourcedataset)
     st = SlideSampler(targetdataset)

     sourceloader = DataLoader(sourcedataset, batch_size = 128, num_workers = 2, sampler = ss)
     targetloader = DataLoader(targetdataset, batch_size = 128, num_workers = 2, sampler = st)

     valloader = DataLoader(valdataset, batch_size = 32, shuffle = False, num_workers = 2)
     testloader = DataLoader(testdataset, batch_size = 32, shuffle = False, num_workers = 2)
     testloader2 = DataLoader(testdataset2, batch_size = 32, shuffle = False, num_workers = 2)

     train_labs, train_counts = np.unique(sourcedataset.labels, return_counts=True)
     
     #defining multi-class weights of the weighted cross-entropy loss function (model gets penalized more for classification error on rare classes)
     weights = np.array([1.1,1.1,1.1,1.0,1.0,1,1])
     train_labs2, train_counts2 = np.unique(targetdataset.labels, return_counts=True)
     
     val_labs, val_counts = np.unique(valdataset.labels, return_counts=True)
     test_labs, test_counts = np.unique(testdataset.labels, return_counts=True)
     print([sourcedataset.classes[l] for l in train_labs])
     print(train_counts)

     print([targetdataset.classes[l] for l in train_labs2])
     print(train_counts2)

     print(weights)
     print([valdataset.classes[l] for l in val_labs])
     print(val_counts)
     print([testdataset.classes[l] for l in test_labs])
     print(test_counts)




     lr = 1e-5

     print("Base params to learn:")
     params_to_update = []
     for name,param in base.named_parameters():
          if param.requires_grad == True:
               params_to_update.append(param)

     optimizer = torch.optim.Adam(params_to_update + list(os_classifier.parameters()) + list(domain_classifier.parameters()), lr, weight_decay=5e-4)


     ###########################################################
     ### Setting the loss function here ###
     ###########################################################
     class_weights = torch.FloatTensor(weights).cuda()
     loss_fn = torch.nn.CrossEntropyLoss(weight = class_weights)



     dataset_dict = {"train": valdataset, "val": testdataset}
     epoch = 0
     val_r = []
     val_p = []
     val_best_f1 = 0.0
     val_best_test_f1 = 0.0
     best_f1 = 0.0
    
     test_r = []
     test_p = []
     test2_r = []
     test2_p = []
     train_losses = []
     val_losses = []
     test_losses = []
     test2_losses = []
     val_losses_sds = []
     test_losses_sds = []
     test2_losses_sds = []

     val_precisions = []
     val_recalls = []
     test_precisions = []
     test_recalls = []
     test2_precisions = []
     test2_recalls = []

     num_epochs = 15
     for epoch in range(1, num_epochs+1):
          lambd = 0.0
          #defining the adversarial learning parameter (lambda) annealing schedule (Ganin & Lempinsky et al, 2014)
          if epoch > 0:
               if (epoch-1)%3 == 0:
                    lambd = 0.0
               else:
                    lambd = (2. / (1. + np.exp(-10 * float((epoch-1))/num_epochs)) - 1)
               
          domain_classifier.set_lambda(lambd)

          #defining the learning rate update schedule (Ganin & Lempinsky et al, 2014)
          lr = 1e-3 / (1. + 10 * float(epoch)/num_epochs)**0.75
          optimizer.lr = lr
          train_losses.append(train_supervised(base, os_classifier, domain_classifier, loss_fn, optimizer, sourceloader, targetloader))
          
          val_loss, val_precision, val_recall, val_f1, vper_class_prec, vper_class_recall, vper_class_f1, val_elosses = test(base, os_classifier, valloader, loss_fn, epoch, valdataset.classes)
          test_loss, test_precision, test_recall, test_f1, tper_class_prec, tper_class_recall, tper_class_f1, test_elosses = test(base, os_classifier, testloader, loss_fn, epoch, testdataset.classes)
          test2_loss, test2_precision, test2_recall, test2_f1, t2per_class_prec, t2per_class_recall, t2per_class_f1, test2_elosses = test(base, os_classifier, testloader2, loss_fn, epoch, testdataset2.classes)
          val_losses.append(val_loss)
          test_losses.append(test_loss)
          test2_losses.append(test2_loss)
          val_losses_sds.append(np.std(val_elosses))
          test_losses_sds.append(np.std(test_elosses))
          test2_losses_sds.append(np.std(test2_elosses))
          val_r.append(val_recall)
          val_p.append(val_precision)
          test_r.append(test_recall)
          test_p.append(test_precision)
          test2_r.append(test2_recall)
          test2_p.append(test2_precision)
          val_precisions.append(vper_class_prec)
          val_recalls.append(vper_class_recall)
          test_precisions.append(tper_class_prec)
          test_recalls.append(tper_class_recall)
          test2_precisions.append(t2per_class_prec)
          test2_recalls.append(t2per_class_recall)

          #save model achieving the best classification performance on the validation dataset only.
          if val_f1 > best_f1:
               best_f1 = val_f1
               val_best_f1 = val_f1
               val_best_test_f1 = test2_f1
               save_model(base, 'resnet50_allclasses_10x_domain_adapt_final_fe_033122{}'.format(level0))
               save_model(os_classifier, 'resnet50_allclasses_10x_domain_adapt_final_os_classifier_033122{}'.format(level0))


     with open('./resnet50_allclasses_10x_domain_adapt_train_losses_final{}'.format(level0),'wb') as f:
          pickle.dump(train_losses,f)

     with open('./resnet50_allclasses_10x_domain_adapt_val_losses_final{}'.format(level0),'wb') as f:
          pickle.dump(val_losses,f)

     with open('./resnet50_allclasses_10x_domain_adapt_test_losses_final{}'.format(level0),'wb') as f:
          pickle.dump(test_losses,f)

     with open('./resnet50_allclasses_10x_domain_adapt_test2_losses_final{}'.format(level0),'wb') as f:
          pickle.dump(test2_losses,f)

     with open('./resnet50_allclasses_10x_domain_adapt_val_losses_sds_final{}'.format(level0),'wb') as f:
          pickle.dump(val_losses_sds,f)

     with open('./resnet50_allclasses_10x_domain_adapt_test_losses_sds_final{}'.format(level0),'wb') as f:
          pickle.dump(test_losses_sds,f)

     with open('./resnet50_allclasses_10x_domain_adapt_test2_losses_sds_final{}'.format(level0),'wb') as f:
          pickle.dump(test2_losses_sds,f)

     with open('./resnet50_allclasses_10x_domain_adapt_val_precisions_final{}'.format(level0),'wb') as f:
          pickle.dump(val_precisions,f)

     with open('./resnet50_allclasses_10x_domain_adapt_val_recalls_final{}'.format(level0),'wb') as f:
          pickle.dump(val_recalls,f)

     with open('./resnet50_allclasses_10x_domain_adapt_test_precisions_final{}'.format(level0),'wb') as f:
          pickle.dump(test_precisions,f)

     with open('./resnet50_allclasses_10x_domain_adapt_test_recalls_final{}'.format(level0),'wb') as f:
          pickle.dump(test_recalls,f)

     ep = range(1,num_epochs+1)
     plt.xticks(np.arange(1,num_epochs+1))
     plt.plot(ep, val_r, 'g', label='validation set mean Recall')
     plt.plot(ep, val_p, 'b', label='validation set mean Precision')
     plt.title('best val mean F1:{:.2f}, test mean F1:{:.2f}'.format(val_best_f1, val_best_test_f1))
     plt.xlabel('epochs')
     plt.ylabel('performance')
     plt.legend()
     plt.show()
     plt.savefig('{}_{}_{}_model_perf.png'.format('resnet50_allclasses_10x_domain_adapt__final','mlp',level0))
     plt.close()


     plt.xticks(np.arange(1,num_epochs+1))
     plt.plot(ep, train_losses, 'g', label='trainset set loss')
     plt.plot(ep, val_losses, 'b', label='validation set loss (dog)')
     plt.plot(ep, test_losses, 'r', label = 'test set loss (human)')
     plt.title('Train and validation set losses')
     plt.xlabel('epochs')
     plt.ylabel('loss')
     plt.legend()
     plt.show()
     plt.savefig('{}_{}_{}_model_losses.png'.format('resnet50_allclasses_10x_domain_adapt__final_033122','mlp',level0))
     plt.close()
     res = pd.DataFrame({'epoch': ep, 'train loss': train_losses, 'val loss': val_losses,
                  'val mean precision': val_p, 'val mean recall': val_r, 'test mean precision':  test_p, 'test mean recall':test_r})
     res.to_csv('{}_{}_{}_results.csv'.format('resnet50_allclasses_10x_domain_adapt__final','mlp',level0))




