# SAR-ATR
Implementation of "EM Simulation-Aided Zero-Shot Learning for SAR Automatic Target Recognition".

## System requirements
- Windows 7
- Python 3.6+/2.7+
- Tensorflow
- SciPy
- NumPy
- MATLAB

## Datasets
A few MSTAR samples and simulated T72 data are listed here. MSTAR data can be downloaded from https://www.sdms.afrl.af.mil/-datasets/mstar/. 
- Synthetic image: HB06165_with_mstar.mat
- Synthetic image: HB06181_with_mstar.mat
- Detection network: download it at [here](https://pan.baidu.com/s/16qa5H2ROaJTg3zfAOm63Kg) and store it in './data/'.
- VGG19 model: download it at [here](https://pan.baidu.com/s/1nJTTjmZIsneTgv_Uf8DxgA) and store it in './data/'.

## Target detection
Use `target_detection` to detect the targets in the synthetic image. It read in HB06165_with_mstar.mat, detect the targets, modify the cropped images and save it to ./data/scene_test_181_fans.mat.

## Target classification in Python
Run `Python main.py --is_test=True` to classify the detected images. The classification result wil be saved at ./result/pred.mat.

## Show the classification results
Use `target_cla` to compare the ground truth and recognition results.

## Others
- To train the network: `Python main.py --mode='ZSL'`
- Ablation experiments: `Python main.py --mode='-FANS'` `Python main.py --mode='-Style'` `Python main.py --mode='-Segmentation'`

## Author
- [Qian Song](https://github.com/QianSong-Cherry)  Contact me at songq15@fudan.edu.cn.
- Qian Guo
- Wei Ao


## Reference
[1] Q. Song, H. Chen, F. Xu, and T.J. Cui, "EM Simulation-Aided Zero-Shot Learning for SAR Automatic Target Recognition," IEEE GRSL, 2019.
[2] D. Cozzolino, S. Parrilli, G. Scarpa, G. Poggi and L. Verdoliva. “Fast Adaptive Nonlocal SAR Despeckling,” IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 2, pp. 524-528, 2014.
[3]	K. Simonyan, and A. Zisserman, “Very Deep Convolutional Networks for Large-scale Image Recognition,” arXiv:1409.1556, 2014.
[4] L. J. P. van der Maaten and G. E. Hinton, “Visualizing High-Dimensional Data Using t-SNE,” Journal of Machine Learning Research, vol. 9, pp. 2579-2605, 2008.
