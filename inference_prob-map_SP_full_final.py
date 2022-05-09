import sys
import os
import glob
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["OMP_NUM_THREADS"] = "10"
import numpy as np
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import geojson
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from torchvision import models
#from fastai.vision.all import *
import matplotlib.pyplot as plt
#import fastai
import PIL
import matplotlib
matplotlib.use('Agg')
import pandas as pd
# import staintools
import datetime
from skimage import draw, measure, morphology
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
from tqdm import tqdm
import glob
warnings.filterwarnings("ignore")

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

class extractPatch:

    def __init__(self, fe_path, classifier_path):
        self.save_location = '/data/spatkar/slide_probs_dog'#args.save_dir #'/data/AIR/Chen_AI/Data/json_output'
        self.mag_extract = [10]#[int(args.mag)] # specify which magnifications you wish to pull images from
        self.save_image_size = 256   # specify image size to be saved (note this is the same for all magnifications)
        self.run_image_size = 256
        self.pixel_overlap = 64       # specify the level of pixel overlap in your saved images
        self.limit_bounds = True     # this is weird, dont change it
        self.write_all = False       # default is to only write patches that overlap with xml regions (if no xml provided, all patches written)
        self.nolabel = False         # if all regions in an annotation file belong to the same class, they are labeled as 'tumor'
                                     #      nolabel=FALSE should only be used if the "Text" attribute in xml corresponds to label

        

        self.model0_path = fe_path
        self.model1_path = classifier_path #args.model_file 
        
        self.stain = 'none' #args.normstain # should be 'none', 'macenko' or 'vahadane'
        
    def parseMeta_and_pullTiles(self, csvfile):
        if not os.path.exists(os.path.join(self.save_location)):
            os.mkdir(os.path.join(self.save_location))

        #first load pytorch model
        #learn = load_learner(self.model_path,cpu=False)
        base = FeatureExtractor(backbone = 'resnet50', use_pretrained = False, freeze_weights=False)
        os_classifier = Classifier(indim = base.out_dim, num_classes = 7, hidden_layers = 0)
        
        base.load_state_dict(torch.load(self.model0_path))
        os_classifier.load_state_dict(torch.load(self.model1_path))
        

        base = base.cuda()
        os_classifier = os_classifier.cuda()
        

        base.eval()
        os_classifier.eval()
        

        
        filelist = list(pd.read_csv(csvfile).files.values)
        done_files = []
        slides_done = list(np.unique([x.split('_')[2].split('/')[1] for x in done_files]))
        normalize = transforms.Normalize(mean=[0.8938, 0.5708, 0.7944], std = [0.1163, 0.1528, 0.0885])
        patchtransform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize])
        for _file in tqdm(filelist):

            # first grab data from digital header
            # print(os.path.join(self.file_location,self.image_file))
            #oslide = openslide.OpenSlide(os.path.join(self.file_location,self.image_file))
            #'2021-04-23 10.59.46','2021-04-22 17.27.42','2021-04-24 09.41.08','2021-04-22 21.00.02','2021-04-22 17.02.20','2021-04-22 18.50.37','2021-04-22 16.45.37'
            savnm = os.path.basename(_file)
            self.save_name = str.replace(savnm,'.ndpi','')
            print(self.save_name)
            if self.save_name in slides_done:
                continue
            oslide = openslide.OpenSlide(_file)
            print(_file)

            #mrxs files have an offset
            #offset = [int(oslide.properties[openslide.PROPERTY_NAME_BOUNDS_X]),
            #    int(oslide.properties[openslide.PROPERTY_NAME_BOUNDS_Y])]

            # this is physical microns per pixel
            acq_mag = 10.0/float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])

            # this is nearest multiple of 20 for base layer
            base_mag = int(20 * round(float(acq_mag) / 20))

            # this is how much we need to resample our physical patches for uniformity across studies
            physSize = round(self.save_image_size*acq_mag/base_mag)

            # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
            tiles = DeepZoomGenerator(oslide, tile_size=physSize-round(self.pixel_overlap*acq_mag/base_mag), overlap=round(self.pixel_overlap*acq_mag/base_mag/2), limit_bounds=self.limit_bounds)

            # calculate the effective magnification at each level of tiles, determined from base magnification
            tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))


            # intermeadiate level for probability map
            # in my case, I selected level 5 because it was a reasonable size (10000x10000 pixels) to see regional characteristics without being too large
            lvl_img = oslide.read_region((0,0),5,oslide.level_dimensions[5])
            # x_map is where I save probabilities... 
            # PIL image flips x and y coords
            #OB vs rest
            x_count = np.zeros((lvl_img.size[1],lvl_img.size[0]),  float)
            
            x_mapOB = np.zeros((lvl_img.size[1],lvl_img.size[0]),  float)

            #HN vs rest
            x_mapHN = np.zeros((lvl_img.size[1],lvl_img.size[0]),  float)

            #CB vs rest
            x_mapCB = np.zeros((lvl_img.size[1],lvl_img.size[0]),  float)

            #FB vs rest
            x_mapFB = np.zeros((lvl_img.size[1],lvl_img.size[0]),  float)

            #GC vs rest
            x_mapGC = np.zeros((lvl_img.size[1],lvl_img.size[0]),  float)

            #VR vs rest
            x_mapVR = np.zeros((lvl_img.size[1],lvl_img.size[0]),  float)

            # this is the resize from original size, so we can re-create the probability in low map
            lvl_resize = oslide.level_downsamples[5]

            # pull tiles from levels specified by self.mag_extract
            for lvl in self.mag_extract:
                if lvl in tile_lvls:
                    # print(lvl)
                    # pull tile info for level
                    x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]

                    for y in range(0,y_tiles):
                        for x in range(0,x_tiles):

                            # grab tile coordinates
                            tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                            save_coords = str(tile_coords[0][0]) + "-" + str(tile_coords[0][1]) + "_" + '%.0f'%(tiles._l0_l_downsamples[tile_coords[1]]*tile_coords[2][0]) + "-" + '%.0f'%(tiles._l0_l_downsamples[tile_coords[1]]*tile_coords[2][1])
                            tile_ends = (int(tile_coords[0][0] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]),int(tile_coords[0][1] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1]))
                            tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                            ws = self.whitespace_check(im=tile_pull)
                            if ws < 0.85:
                                tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.ANTIALIAS)
                                tile_pull = tile_pull.resize(size=(self.run_image_size, self.run_image_size),
                                                             resample=PIL.Image.ANTIALIAS)
                                tile_pull = patchtransform(tile_pull).reshape(-1,3,224,224)
                                #print(tile_pull.size())

                                features = base(tile_pull.cuda())
                                label_preds = os_classifier(features)
                                

                                labs = ['CB','FB','GC','N','VR','other','OB']
                                probs = torch.softmax(label_preds, dim = 1).detach().cpu().numpy()[0]
                                

                                #probs = probs/np.sum(probs)

                                #print(outputs_np)
                                # super simple allocation
                                x_count[int(np.floor(tile_coords[0][1]/lvl_resize)):int(np.floor(tile_ends[1]/lvl_resize)),int(np.floor(tile_coords[0][0]/lvl_resize)):int(np.floor(tile_ends[0]/lvl_resize))] += 1
                                x_mapOB[int(np.floor(tile_coords[0][1]/lvl_resize)):int(np.floor(tile_ends[1]/lvl_resize)),int(np.floor(tile_coords[0][0]/lvl_resize)):int(np.floor(tile_ends[0]/lvl_resize))] += probs[6]
                                x_mapHN[int(np.floor(tile_coords[0][1]/lvl_resize)):int(np.floor(tile_ends[1]/lvl_resize)),int(np.floor(tile_coords[0][0]/lvl_resize)):int(np.floor(tile_ends[0]/lvl_resize))] += probs[3]
                                x_mapCB[int(np.floor(tile_coords[0][1]/lvl_resize)):int(np.floor(tile_ends[1]/lvl_resize)),int(np.floor(tile_coords[0][0]/lvl_resize)):int(np.floor(tile_ends[0]/lvl_resize))] += probs[0]
                                x_mapFB[int(np.floor(tile_coords[0][1]/lvl_resize)):int(np.floor(tile_ends[1]/lvl_resize)),int(np.floor(tile_coords[0][0]/lvl_resize)):int(np.floor(tile_ends[0]/lvl_resize))] += probs[1]
                                x_mapGC[int(np.floor(tile_coords[0][1]/lvl_resize)):int(np.floor(tile_ends[1]/lvl_resize)),int(np.floor(tile_coords[0][0]/lvl_resize)):int(np.floor(tile_ends[0]/lvl_resize))] += probs[2]
                                x_mapVR[int(np.floor(tile_coords[0][1]/lvl_resize)):int(np.floor(tile_ends[1]/lvl_resize)),int(np.floor(tile_coords[0][0]/lvl_resize)):int(np.floor(tile_ends[0]/lvl_resize))] += probs[4]
                                
                                

                else:
                    print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")

            # divide by number of counts
            x_count = np.where(x_count < 1, 1, x_count)
            x_mapOB = x_mapOB / x_count
            x_mapHN = x_mapHN / x_count
            x_mapCB = x_mapCB / x_count
            x_mapFB = x_mapFB / x_count
            x_mapGC = x_mapGC / x_count
            x_mapVR = x_mapVR / x_count
            # i save in black and white and in color
            # original img
            lvl_img.save(os.path.join(self.save_location,self.save_name+'_lowres.tiff'))


            # for OB
            slideimgOB = PIL.Image.fromarray(np.uint8(x_mapOB * 255))
            #np.savetxt(os.path.join(self.save_location,self.save_name+'_cancer_probOB.txt'), x_mapOB, delimiter=',')
            slideimgOB = slideimgOB.convert('L')
            slideimgOB.save(os.path.join(self.save_location,self.save_name+'_cancer_probOB.jpeg'))

            cmap = plt.get_cmap('jet')
            rgba_imgOB = cmap(x_mapOB)
            rgb_imgOB = np.delete(rgba_imgOB, 3, 2)
            colimgOB = PIL.Image.fromarray(np.uint8(rgb_imgOB * 255))
            colimgOB.save(os.path.join(self.save_location,self.save_name+'_cancer_colorOB.jpeg'))


            # for HN
            slideimgHN = PIL.Image.fromarray(np.uint8(x_mapHN * 255))
            #np.savetxt(os.path.join(self.save_location,self.save_name+'_cancer_probHN.txt'), x_mapHN, delimiter=',')
            slideimgHN = slideimgHN.convert('L')
            slideimgHN.save(os.path.join(self.save_location,self.save_name+'_cancer_probHN.jpeg'))

            cmap = plt.get_cmap('jet')
            rgba_imgHN = cmap(x_mapHN)
            rgb_imgHN = np.delete(rgba_imgHN, 3, 2)
            colimgHN = PIL.Image.fromarray(np.uint8(rgb_imgHN * 255))
            colimgHN.save(os.path.join(self.save_location,self.save_name+'_cancer_colorHN.jpeg'))


            # for CB
            slideimgCB = PIL.Image.fromarray(np.uint8(x_mapCB * 255))
            #np.savetxt(os.path.join(self.save_location,self.save_name+'_cancer_probCB.txt'), x_mapCB, delimiter=',')
            slideimgCB = slideimgCB.convert('L')
            slideimgCB.save(os.path.join(self.save_location,self.save_name+'_cancer_probCB.jpeg'))

            cmap = plt.get_cmap('jet')
            rgba_imgCB = cmap(x_mapCB)
            rgb_imgCB = np.delete(rgba_imgCB, 3, 2)
            colimgCB = PIL.Image.fromarray(np.uint8(rgb_imgCB * 255))
            colimgCB.save(os.path.join(self.save_location,self.save_name+'_cancer_colorCB.jpeg'))


            # for FB
            slideimgFB = PIL.Image.fromarray(np.uint8(x_mapFB * 255))
            #np.savetxt(os.path.join(self.save_location,self.save_name+'_cancer_probFB.txt'), x_mapFB, delimiter=',')
            slideimgFB = slideimgFB.convert('L')
            slideimgFB.save(os.path.join(self.save_location,self.save_name+'_cancer_probFB.jpeg'))

            cmap = plt.get_cmap('jet')
            rgba_imgFB = cmap(x_mapFB)
            rgb_imgFB = np.delete(rgba_imgFB, 3, 2)
            colimgFB = PIL.Image.fromarray(np.uint8(rgb_imgFB * 255))
            colimgFB.save(os.path.join(self.save_location,self.save_name+'_cancer_colorFB.jpeg'))

            # for GC
            slideimgGC = PIL.Image.fromarray(np.uint8(x_mapGC * 255))
            #np.savetxt(os.path.join(self.save_location,self.save_name+'_cancer_probGC.txt'), x_mapGC, delimiter=',')
            slideimgGC = slideimgGC.convert('L')
            slideimgGC.save(os.path.join(self.save_location,self.save_name+'_cancer_probGC.jpeg'))

            cmap = plt.get_cmap('jet')
            rgba_imgGC = cmap(x_mapGC)
            rgb_imgGC = np.delete(rgba_imgGC, 3, 2)
            colimgGC = PIL.Image.fromarray(np.uint8(rgb_imgGC * 255))
            colimgGC.save(os.path.join(self.save_location,self.save_name+'_cancer_colorGC.jpeg'))


            # for VR
            slideimgVR = PIL.Image.fromarray(np.uint8(x_mapVR * 255))
            #np.savetxt(os.path.join(self.save_location,self.save_name+'_cancer_probVR.txt'), x_mapVR, delimiter=',')
            slideimgVR = slideimgVR.convert('L')
            slideimgVR.save(os.path.join(self.save_location,self.save_name+'_cancer_probVR.jpeg'))

            cmap = plt.get_cmap('jet')
            rgba_imgVR = cmap(x_mapVR)
            rgb_imgVR = np.delete(rgba_imgVR, 3, 2)
            colimgVR = PIL.Image.fromarray(np.uint8(rgb_imgVR * 255))
            colimgVR.save(os.path.join(self.save_location,self.save_name+'_cancer_colorVR.jpeg'))

        return

    def whitespace_check(self,im):
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw=bw/255
        prop_ws = (bw > 0.8).sum()/(bw>0).sum()
        return prop_ws

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="csv file containing paths to each whole slide image")
    parser.add_argument("--fe", type=str, help="path to file storing trained weights of resnet50 feature extractor")
    parser.add_argument("--cls", type=str, help="path to file storing trained weights of histological subtype classifier")
    args = parser.parse_args()
    CSV_FILE = args.csv
    FE_FILE = args.fe
    CLS_FILE = args.cls
    c = extractPatch(FE_FILE, CLS_FILE)
    c.parseMeta_and_pullTiles(CSV_FILE)

