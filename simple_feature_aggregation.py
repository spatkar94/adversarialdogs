# Contributor: Sushant Patkar
# Email: patkar.sushant@nih.gov
# Artificial Intelligence Resource, National Cancer Institute
# Sept 30, 2022
# THIS SOFTWARE IS PROVIDED BY THE CONTRIBUTORS "AS IS" FOR THE PURPOSES OF ACADEMIC RESEARCH 
# ONLY AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
# OF THE POSSIBILITY OF SUCH DAMAGE.

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
        parser.add_argument("--savedir", type=str, help="path to save location")
        args = parser.parse_args()
	SAVE_DIR = args.savedir
	features = {}
	imgs = sorted(glob.glob(os.path.join(SAVE_DIR,'*_lowres.tiff')))
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
		probmaps = sorted(glob.glob(os.path.join(SAVE_DIR,'*_prob{}.jpeg'.format(lab))))
		print(probmaps[:10])
		slides = [os.path.basename(x).split('_')[0] for x in sorted(glob.glob('/data/spatkar/slide_probs_dog/*_prob{}.jpeg'.format(lab)))]
		features[lab] = {}
		for i in tqdm(range(len(probmaps))):
			map_ = probmaps[i]
			img = Image.open(map_).convert('L')
			img = np.array(img)/255.0
			features[lab][slides[i]] = (np.sum(img > 0.5))

	slide_features = pd.DataFrame(features)
	slide_features.to_csv('dog_slide_level_features.csv')



		


