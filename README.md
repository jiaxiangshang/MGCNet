# Self-Supervised Monocular 3D Face Reconstruction by Occlusion-Aware Multi-view Geometry Consistency(ECCV 2020, unfinish, try to finish before August 15)
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
```bash
git clone --recursive https://github.com/jiaxiangshang/MGCNet.git
cd MGCNet
(sudo) pip install -r requirement.txt
```
The thirdlib(diff render) is from https://github.com/google/tf_mesh_renderer.
I fork and make changes, and the setting is bazel==10.1, gcc==5.4, the command is bazel build ...

## 2.Model
1. 3dmm + network weight
  https://drive.google.com/file/d/1RkTgcSGNs2VglHriDnyr6ZS5pbnZrUnV/view?usp=sharing
  Extrack this file to /MGCNet/model
2. pretain
  https://drive.google.com/file/d/1jVlf05_Bm_nbIQXZRfmz-dA03xGCawBw/view?usp=sharing
  Extrack this file to /MGCNet/pretain
  
## 3.Data
1. data
  https://drive.google.com/file/d/1Du3iRO0GNncZsbK4K5sboSeCUv0-SnRV/view?usp=sharing
  Extrack this file to /MGCNet/data
  (We can not provide all data as it is too large and the lisence of MPIE dataset[http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html]).
2. data: landmark ground truth(https://github.com/1adrianb/2D-and-3D-face-alignment)
   We use the SFD face detector
3. data: skin prob
  I get the code from Yu DENG(t-yudeng@microsoft.com), maybe your can ask help from him.

## 4.Testing
1. test_image.py(unfinish)
  This is used to inference single unprocessed image(cmd in file)
2. test_prepro_folder.py(unfinish)
  Test all the images in a folder which processed by face detection and face alignment.
  This file can also render the images(geometry,texture,shading,multi-pose), like above or in our paper(read code).
  
## 5.Training
1. train_unsupervise.py(finish)

# Useful tools
1. 3D face render(unfinish)ing for comparison.
2. Build aligned face for your face model.

# Code structure
This part aim at that you can read the code easily.(I will also comment in the code)
1. data structure
2. logic

# Citation
If you use this code, please consider citing:

```
eccv
```

# Contacts
Please contact _jiaxiang.shang@gmail.com_  or open an issue for any questions or suggestions.

## Acknowledgements
