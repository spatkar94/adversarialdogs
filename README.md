# Deep domain adversarial learning for classification of rare histological variants of osteosarcomas in canines and humans
Osteosarcomas are aggressive tumors of the bone with many divergent histologies. High tumor heterogeneity coupled with scarcity of samples render clinical interpretation and prognosis of osteosarcomas challenging. While osteosarcomas are relatively rare in humans, they are similar to those commonly observed in dogs, both at the histological and molecular level. Here, we apply a domain adversarial learning framework that trains neural networks to distinguish different rare histological subtypes of osteosarcomas in both dogs and humans. We show that adversarial learning improves domain adaption of the classification model from dogs to humans when evaluated on unseen whole slide image patches achieving an average multi-class F1 score of 0.77 (CI: 0.74-0.79) and 0.80 (CI: 0.78-0.81), compared to the ground truth in dogs and humans, respectively. Furthermore, we uncover two distinct populations of dogs based on model-predicted spatial distribution of different histological subtypes, which have markedly different responses to standard of care therapy. This repository contains the codes to train a CNN to classify rare histological variants of osteosarcoma in a species agnostic fashion with the help of [adversarial learning](http://proceedings.mlr.press/v37/ganin15.html). The schematic diagram describing the adversarial learning approach is shown below. See [Survival analysis](Survival_analysis.md) for reproducing survival results. 
![](schematic_adversarial.png)

## Requirements
<ul>
  <li> python>=3.6 </li>
  <li> openslide=1.1.1 </li>
  <li> pytorch=1.9+cu111 </li>
  <li> torchvision=0.10.0+cu111 </li>
</ul>

## Instructions for adversarially training a resnet50 CNN on canine and human whole slide imaging (WSI) data
Prior to running the [training script](dogOS_domain_adapt.py), pre-process the canine (source) and human (target) whole slide imaging cohorts to extract non-overlapping patches from tissue regions. Each WSI patch should be saved in a directory corresponding to the slide with the patch class label appended to the filename. The file structure for each cohort is as follows.

```bash
├── rootdir
│   ├── slide1
│   │   ├── slide1_patch1_patchlabel-%area.png
│   │   ├── slide1_patch2_patchlabel-%area.png
│   │   ├── ...
│   ├── slide2
│   │   ├── slide2_patch1_patchlabel-%area.png
│   │   ├── slide2_patch2_patchlabel-%area.png
│   │   ├── ...
...
...
│   ├── slideN
```
*Note:* A patch can have more than one class label depending on overlapping pathologist annotations. See the scripts for how the class label for such patches is resolved. After pre-processing the data, the training script can be invoked as follows:

```
python dogOS_domain_adapt.py --source <path/to/source/rootdir> --target <path/to/target/rootdir>
```
