# Dynamic-weighting Hierarchical Segmentation Network for Medical Images

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/).

## Summary:

### Intoduction:
This repository is for our MedIA paper ["Dynamic-weighting Hierarchical Segmentation Network for Medical Images"](https://www.sciencedirect.com/science/article/abs/pii/S1361841521002413)

### Framework:
![](https://github.com/CityU-AIM-Group/DW-HieraSeg/blob/main/Figs/network.png)

## Usage:
### Requirement:
Pytorch 1.3
Python 3.6

### Preprocessing:
Clone the repository:
```
git clone https://github.com/CityU-AIM-Group/DW-HieraSeg.git
cd DW-HieraSeg 
bash sh_hierasegCVC.sh
bash sh_dw_hierasegCVC.sh
bash sh_hierasegISIC.sh
bash sh_dw_hierasegISIC.sh
```

### Data preparation:
Dataset should be put into the folder './data'. For example, if the name of dataset is CVC, then the path of dataset should be './data/CVC/', and the folder structure is as following.
```
ThresholdNet
|-data
|--CVC
|---images
|---labels
|---train.txt
|---test.txt
|---valid.txt
```
The content of 'train.txt', 'test.txt' and 'valid.txt' should be just like:
```
26.png
27.png
28.png
...
```

### Pretrained model:
You should download the pretrained model from [Google Drive](https://drive.google.com/file/d/1yeZxwV6dYHQJmj2i5x9PnB6u-rqvlkCj/view?usp=sharing), and then put it in the './model' folder for initialization. 

### Well trained model:
You could download the trained model from [Google Drive](https://drive.google.com/file/d/1YjwOYYJxl-Vv3UEgblZaExPgLK74RvI6/view?usp=sharing), which achieves 82.328% in Jaccard score on the [EndoScene testing dataset](https://www.hindawi.com/journals/jhe/2017/4037190/). Put the model in directory './models'.

## Citation:
```
@article{guo2021dynamic,
  title={Dynamic-weighting Hierarchical Segmentation Network for Medical Images},
  author={Guo, Xiaoqing and Yang, Chen, Yuan, Yixuan},
  journal={Medical Image Analysis},
  year={2021}
}
```

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
