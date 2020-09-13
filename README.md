# Self-Supervised Monocular 3D Face Reconstruction by Occlusion-Aware Multi-view Geometry Consistency(ECCV 2020)
This is an official python implementation of MGCNet. This is the pre-print version https://arxiv.org/abs/2007.12494.

# Result
1. video
  <p align="center"> 
  <img src="githubVisual/ECCV2020_Github.gif">
  </p>
  
2. image
  ![image](https://github.com/jiaxiangshang/MGCNet/blob/master/githubVisual/result_multiPose.jpg)
  
3. Full video can be seen in [YouTube] https://www.youtube.com/watch?v=DXzkO3OwlYQ
  
# Running code
## 1. Code + Requirement + thirdlib
We run this with python3.7
```bash
git clone --recursive https://github.com/jiaxiangshang/MGCNet.git
cd MGCNet
(sudo) pip install -r requirement.txt
```
(1) The thirdlib(diff render) is from https://github.com/google/tf_mesh_renderer. Many issue happens here, so let's make this more clear.
This diff render does not return triangle id for each pixel after raterise, we do this by our self and add these changes as submodule to this github. 

(2) Then how to compile, the setting is bazel==10.1, gcc==5.*, the command is 
```bash
bazel build ...
```
The gcc/g++ version higher than 5.* will bring problems, a good solution is virtual environment with a gcc maybe 5.5.
If the The gcc/g++ version is 4.*, you can try to change the compile cmd in BUILD file, about the flag -D_GLIBCXX_USE_CXX11_ABI=0 or -D_GLIBCXX_USE_CXX11_ABI=1
Which is used for new cxx11 property for gcc5.*

## 2.Model
1. 3dmm model + network weight(I will check and update on August 21)
  https://drive.google.com/file/d/1RkTgcSGNs2VglHriDnyr6ZS5pbnZrUnV/view?usp=sharing
  Extract this file to /MGCNet/model
2. pretain
  https://drive.google.com/file/d/1jVlf05_Bm_nbIQXZRfmz-dA03xGCawBw/view?usp=sharing
  Extract this file to /MGCNet/pretain
  
## 3.Data
1. data
  https://drive.google.com/file/d/1Du3iRO0GNncZsbK4K5sboSeCUv0-SnRV/view?usp=sharing
  Extract this file to /MGCNet/data
  (We can not provide all data as it is too large and the license of MPIE dataset[http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html]).
2. data: landmark ground truth(https://github.com/1adrianb/2D-and-3D-face-alignment)
   We use the SFD face detector
3. data: skin prob
  I get the code from Yu DENG(t-yudeng@microsoft.com), maybe you can ask help from him.

## 4.Testing
1. test_image.py
  This is used to inference a single unprocessed image(cmd in file).
  This file can also render the images(geometry, texture, shading,multi-pose), like above or in our paper(read code), which makes visualization and comparison more convenient.
2. preprocess
  All the preprocess has been included in 'test_image.py', we show the outline here.
  (1) face detection and face alignment are package in ./tools/preprocess/detect_landmark,py.
  (2) face alignment by affine transformation to warp the unprocess image.
  Test all the images in a folder can follow this preprocess.
  
## 5.Training
1. train_unsupervise.py

# Useful tools(keep updating)
1. face alignment tools
2. 3D face render tools.
3. Camera augment for rendering.

# Citation
If you use this code, please consider citing:

```
@article{shang2020self,
  title={Self-Supervised Monocular 3D Face Reconstruction by Occlusion-Aware Multi-view Geometry Consistency},
  author={Shang, Jiaxiang and Shen, Tianwei and Li, Shiwei and Zhou, Lei and Zhen, Mingmin and Fang, Tian and Quan, Long},
  journal={arXiv preprint arXiv:2007.12494},
  year={2020}
}
```

# Contacts
Please contact _jiaxiang.shang@gmail.com_  or open an issue for any questions or suggestions.

## Acknowledgements
