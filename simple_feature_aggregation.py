import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import cv2
from scipy import ndimage
from skimage import morphology
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
	features = {}
	imgs = sorted(glob.glob('/data/spatkar/slide_probs_dog/*_lowres.tiff'))
	print(imgs[:10])
	tissue_masks = []
	for i in tqdm(range(len(imgs))):
		img = Image.open(imgs[i])
		he = np.array(img)
		he = he[:, :, 0:3]
		heHSV = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY)
		ret, thresh1 = cv2.threshold(heHSV, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		imagem = cv2.bitwise_not(thresh1)
		tissue_mask = morphology.binary_dilation(imagem, morphology.disk(radius=5))
		tissue_mask = morphology.remove_small_objects(tissue_mask, 1000)
		tissue_mask = ndimage.binary_fill_holes(tissue_mask)
		tissue_masks.append(tissue_mask)

	classes = ['OB','CB','FB','GC','HN','VR']
	for lab in classes:
		print(lab)
		probmaps = sorted(glob.glob('/data/spatkar/slide_probs_dog/*_prob{}.jpeg'.format(lab)))
		print(probmaps[:10])
		slides = [os.path.basename(x).split('_')[0] for x in sorted(glob.glob('/data/spatkar/slide_probs_dog/*_prob{}.jpeg'.format(lab)))]
		features[lab] = {}
		for i in tqdm(range(len(probmaps))):
			map_ = probmaps[i]
			img = Image.open(map_).convert('L')
			img = np.array(img)/255.0
			features[lab][slides[i]] = (np.sum(img > 0.5))

	slide_features = pd.DataFrame(features)
	slide_features.to_csv('/data/spatkar/dog_slide_level_features.csv')



		


