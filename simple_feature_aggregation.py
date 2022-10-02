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



		


