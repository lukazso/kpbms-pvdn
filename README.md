# Combining Visual Saliency Methods and Sparse Keypoint Annotations to Providently Detect Vehicles at Night 

The corresponding paper was submitted to ICRA 2023 and is still under review. The 
arxiv preprint can be found [here](https://arxiv.org/abs/2204.11535). This 
repository contains the main source code 
for our context-aware Boolean map saliency approach.

## Setup environment

### PVDN Dataset
Download the PVDN dataset from [Kaggle](https://www.kaggle.com/saralajew/provident-vehicle-detection-at-night-pvdn).

### Python dependencies
The code was tested on Ubuntu 20.04 and Python 3.8.10. However, it should normally 
also work fine on other OS and similar Python 3.x versions.

Setup a virtual environment and install the repo as a package:
```
python3 -m venv venv
source venv/bin/activate
pip install pip --upgrade
pip install -e .
```

## Tutorials
The following notebooks provide examples for basic use cases:
- [Generate the saliency maps](notebooks/generate_saliencymap_dataset.ipynb)
- [Generate the saliency bounding boxes](notebooks/generate_saliencybox_dataset.ipynb)
