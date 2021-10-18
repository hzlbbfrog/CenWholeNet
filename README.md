# CenWholeNet
Automatic damage detection using anchor-free method and unmanned surface vessel
Zhili He, Shang Jiang, Jian Zhang and Gang Wu. Our paper is available.

The codes will be available after the paper is accepted.

# Usage
1. Requirements
    * python >= 3.5
    * pytorch >= 1.1.0
    * CUDA 10.0 and CUDNN 7.4  
    * Other requirements can be found in the requirements.txt.

2. Clone the repo
~~~
git clone https://github.com/hzlbbfrog/CenWholeNet.git 
cd CenWholeNet
~~~
Or, you can "Download ZIP".

3. Compile DCN and nms
You can refer to [CenterNet](https://github.com/xingyizhou/CenterNet/) to compile DCN and nms in advance.

4. Rewrite Damage.py  
For your own data set, you should rewrite **Damage.py**.

5. Train your own data set
* Resnet
~~~
python train.py --log_name Resnet18 --dataset Damage --arch resnet --lr 5e-4 --lr_step 90,120 --batch_size 2 --num_epochs 60 --num_workers 2
~~~
* PAM
~~~
python train_seed.py --log_name Resnet18_PAM --dataset Damage --arch resnet_PAM --lr 5e-4 --lr_step 90,120 --batch_size 2 --num_epochs 60 --num_workers 2
~~~

6. Test your own data set
* Resnet
~~~
python test.py --log_name Resnet18 --arch resnet
~~~
* PAM
~~~
python test.py --log_name Resnet18_PAM --arch resnet_PAM
~~~

# About CenterNet
The official [repo](https://github.com/xingyizhou/CenterNet/) was not adopted because of some reasons.  
A [simple pytorch implementation version](https://github.com/zzzxxxttt/pytorch_simple_CenterNet_45) was used, which is **simpler and easier** to read.  
What's more, a detailed compilation process is introduced in that repo. O(∩_∩)O  
If you want to use CenterNet quickly, **try it**!

# About Faster R-CNN
Faster R-CNN was compared in our paper.  
You can access this [repo](https://github.com/potterhsu/easy-faster-rcnn.pytorch) to get the corresponding codes.  
To tell the truth, it may be a little complicated to compile Faster R-CNN in Win 10. (facepalm)

# About YOLOv5
YOLOv5 was also compared in our paper.  
You can access this [repo](https://https://github.com/ultralytics/yolov5) to get the corresponding codes.

# Citation
You are very welcomed to cite our paper!

# Contact Us
Because I have been busy these days, code is not optimized very well and some notes may be in Chinese.  
I am very sorry for that.  
However, if you have any questions, please do not hesitate to contact me!


