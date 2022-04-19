import sys
import os
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["OMP_NUM_THREADS"] = "10"
import numpy as np
import cv2
import openslide
from PIL import Image
from skimage import draw, measure, morphology, filters
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
from scipy import ndimage
warnings.filterwarnings("ignore")
import glob
import pandas as pd


class convertProb:

    def __init__(self):
        self.file_location = '/home/mip/Mdrive_mount/MIP/AIR/Scanned-images/OS_Beck/Human-osteosarcoma-G-Tom/WSI'
        self.save_location = '/data/spatkar/slide_probs_dog'
        self.mag_extract = [10]#[int(args.mag)] # specify which magnifications you wish to pull images from
        self.save_image_size = 256   # specify image size to be saved (note this is the same for all magnifications)
        self.run_image_size = 1000
        self.pixel_overlap = 0       # specify the level of pixel overlap in your saved images
        self.limit_bounds = True     # this is weird, dont change it
        self.write_all = False       # default is to only write patches that overlap with xml regions (if no xml provided, all patches written)
        self.nolabel = False         # if all regions in an annotation file belong to the same class, they are labeled as 'tumor'
                                     #      nolabel=FALSE should only be used if the "Text" attribute in xml corresponds to label

    def do_mask(self,he_img_path, bw_pred_path, variant):
        he_image = Image.open(he_img_path)
        pred_image = Image.open(bw_pred_path)

        # get he image and find tissue mask
        he = np.array(he_image)
        he = he[:, :, 0:3]
        heHSV = cv2.cvtColor(he, cv2.COLOR_RGB2HSV)
        hetissue = cv2.inRange(heHSV, np.array([135, 10, 30]), np.array([170, 255, 255]))
        hetissue[hetissue > 0] = 1
        hetissue = ndimage.binary_fill_holes(hetissue)
        #if not os.path.exists(str.replace(he_img_path, '_lowres.tiff','_tissue_binary.txt')):
        #    np.savetxt(str.replace(he_img_path, '_lowres.tiff','_tissue_binary.txt'), hetissue, delimiter=',')

        # get pred image
        # I discovered after smoothing my regions tended to lose value due to edges of high probability next to low probability
        # therefore I do a first pass with low prob (0.25) and then later on a filter by region
        preds = np.array(pred_image).astype(np.float)
        preds = preds / 255
        preds[hetissue < 1] = 0
        #preds = filters.gaussian(preds,sigma=50)
        preds_mask = np.zeros(preds.shape)
        preds_mask[preds > 0.5] = 1
        #preds_mask = morphology.binary_dilation(preds_mask, morphology.disk(radius=2))
        #preds_mask = morphology.binary_erosion(preds_mask, morphology.disk(radius=2))
        #preds_mask = morphology.remove_small_objects(preds_mask, 500)
        #preds_mask = ndimage.binary_fill_holes(preds_mask)
        labels = measure.label(preds_mask)
        regions = measure.regionprops(labels,preds)
        for reg in regions:
            # region max must be higher than 0.4
            # this is arbitrary, I did with some experimentation
            if reg.max_intensity<=0.5:
                labels[labels==reg.label]=0
        labels[labels>0]=1
        mask_save = Image.fromarray(np.uint8(labels * 255))
        mask_save = mask_save.convert('L')
        mask_save.save(str.replace(he_img_path, '_lowres.tiff', '_%s_binary.png'%(variant)))
        return labels

    def parseMeta(self):
        filelist = list(pd.read_csv("/data/spatkar/clus2files.csv").file.values)
        slidelist = list(pd.read_csv("/data/spatkar/clus2files.csv").slide.values)
        cmap = {'CB':-16776961, 'FB':-6730752, 'OB': -16777216, 'GC': -6750055, 'HN': -4194112, 'VR':-3670016}
        
        for i in range(len(filelist)):
            _file = filelist[i]
            print(_file)
            json_out = '['
            for variant in ['CB','OB','FB','GC','HN','VR']:
                print(variant)
                he_img_path = _file
                bw_pred_path = str.replace(he_img_path,'_lowres.tiff','_cancer_prob%s.jpeg'%(variant))

                mask_arr = self.do_mask(he_img_path=he_img_path,bw_pred_path=bw_pred_path, variant = variant)
                # first grab data from digital header
                savnm = os.path.basename(_file)
                self.save_name = str.replace(savnm,'_lowres.tiff','')
                 
                #print(slide_location)
                oslide = openslide.OpenSlide(slidelist[i])
                

                # #mrxs files have an offset
                offset = [0, 0]

                # this is physical microns per pixel
                acq_mag = 10.0/float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])

                lvl_resize = oslide.level_downsamples[5]

                polygons = self.tile_ROIS(mask_arr=mask_arr, lvl_resize=lvl_resize)
                if len(polygons) == 0:
                    continue

                json_out += self.slide_ROIS(polygons=polygons, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                    savename='detection', labels=variant, ref=offset, roi_color=cmap[variant]) + ','

            if len(json_out) > 0:
                if json_out[-1] == ',':
                    json_out = json_out[:-1]
            json_out += ']'


            with open(os.path.join(self.save_location,self.save_name+'_qupath' + '.json'), 'w') as outfile:
                outfile.write(json_out)

            print('saved AI polys')

        return

    def whitespace_check(self,im):
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw=bw/255
        prop_ws = (bw > 0.8).sum()/(bw>0).sum()
        return prop_ws


    def slide_ROIS(self,polygons,mpp,savename,labels,ref,roi_color):
        all_polys = unary_union(polygons)
        final_polys = []
        if all_polys.type == 'Polygon':
            poly = all_polys
            polypoints = poly.exterior.xy
            polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
            polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
            # newpoly = Polygon(zip(np.round(polypoints[0],1),np.round(polypoints[1],1)))
            newpoly = Polygon(zip(polyx, polyy))
            if newpoly.area*mpp*mpp > 100:
                final_polys.append(newpoly)

        else:
            for poly in all_polys:
                # print(poly)
                if poly.type == 'Polygon':
                    polypoints = poly.exterior.xy
                    polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                    polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                    # newpoly = Polygon(zip(np.round(polypoints[0],1),np.round(polypoints[1],1)))
                    newpoly = Polygon(zip(polyx, polyy))
                    if newpoly.area*mpp*mpp > 100:
                        final_polys.append(newpoly)
                if poly.type == 'MultiPolygon':
                    for roii in poly.geoms:
                        polypoints = roii.exterior.xy
                        polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                        polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                        # newpoly = Polygon(zip(np.round(polypoints[0], 1), np.round(polypoints[1], 1)))
                        newpoly = Polygon(zip(polyx, polyy))
                        if newpoly.area*mpp*mpp > 100:
                            final_polys.append(newpoly)

        final_shape = unary_union(final_polys)
        try:
            trythis = ''
            for i in range(0, len(final_shape)):
                trythis += json.dumps(
                    {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape[i]),
                    "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                                    "measurements": []}}, indent=4)
                if i < len(final_shape) - 1:
                    trythis += ','
            trythis += ''
        except:
            trythis = ''
            trythis += json.dumps(
                {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape),
                "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                                "measurements": []}}, indent=4)
            trythis += ''

        return trythis

    def tile_ROIS(self,mask_arr,lvl_resize):
        polygons = []
        contours, hier = cv2.findContours(mask_arr.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cvals = contour.transpose(0, 2, 1)
            cvals = np.reshape(cvals, (cvals.shape[0], 2))
            cvals = cvals.astype('float64')
            for i in range(len(cvals)):
                cvals[i][0] = np.round(cvals[i][0]*lvl_resize,2)
                cvals[i][1] = np.round(cvals[i][1]*lvl_resize,2)
            try:
                poly = Polygon(cvals)
                if poly.length > 0:
                    polygons.append(Polygon(poly.exterior))
            except:
                pass

        return polygons

if __name__ == '__main__':
    c = convertProb()
    c.parseMeta()

