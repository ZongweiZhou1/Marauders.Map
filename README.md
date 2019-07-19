## Marauders Map
---

#### Motivation
The motivation for this repo is to visually view the results of trajectory predictions.
For example, results of some trajectory prediction methods, 

[Social LSTM], 

[Social Attention], 

[Social GAN]


#### Dependency packages

- numpy
- matplotlib
- cv2/PIL
- seaborn


#### Data preparation

There are two more commonly used datasets in the field of trajectory prediction.

[ETH](www.vision.ee.ethz.ch/en/datasets/)

Walking pedestrians in busy scenarios from a bird eye view. Manually annotated. 
Data used for training in our ICCV09 paper "You'll Never Walk Alone: Modeling 
Social Behavior for Multi-target Tracking"
```shell
wget www.vision.ee.ethz.ch/datasets_extra/ewap_dataset_full.tgz
```
[UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) 

- Zara Data Set. This dataset contains two video sequence and it can be download as follows:
```buildoutcfg
wget http://graphics.cs.ucy.ac.cy/files/crowds/data/data_zara.rar
wget http://graphics.cs.ucy.ac.cy/files/crowds/data/crowds_zara01.avi
wget http://graphics.cs.ucy.ac.cy/files/crowds/data/crowds_zara02.avi
```
- University Students.
```buildoutcfg
wget http://graphics.cs.ucy.ac.cy/files/crowds/data/data_university_students.rar
wget http://graphics.cs.ucy.ac.cy/files/crowds/data/students003.avi
```
File organization sturcture is as follows:

```buildoutcfg
Marauders.Map
+ ---- data
|      + ----  ETH
|      |       + -----  ewap_dataset
|      |       |        + ----- seq_eth
|      |       |        |       + -----  obsmat.txt
|      |       |        |       + -----  seq_eth.avi
|      |       |        |       + -----  H.txt
|      |       |        |       + -----  groups.txt
|      |       |        |       + -----  destinations.txt
|      |       |        + ----- seq_hotel
|      |       |        |       + -----  obsmat.txt
|      |       |        |       + -----  seq_hotel.avi
|      |       |        |       + -----  H.txt
|      |       |        |       + -----  groups.txt
|      |       |        |       + -----  destinations.txt
|      |       |        + ------README.txt
|      + ----  UCY
|      |       + ----   univ
|      |       |        + ----- students003.vsp
|      |       |        + ----- students003.avi
|      |       |        + ----- ...
|      |       + ----   zara
|      |       |        + ----- zara01
|      |       |        |       + ------ crowds_zara01.avi
|      |       |        |       + ------ crowds_zara01.vsp
|      |       |        + ----- zara02
|      |       |        |       + ------ crowds_zara02.avi
|      |       |        |       + ------ crowds_zara02.vsp
|      |       |        + ----- crowd_file_format.txt
```

The interpolated annotations are saved in `pixel_pos_interpolate.csv` in each corresponding subdir.

#### Run

Here, we provide a tool to extract frames from video, of course, you can directly operate from video without save frames.

```buildoutcfg
python video2frame.py
```

Then you can observation the ground truth trajectories:
```buildoutcfg
python Marauders_Map.py --dataset zara01 --ghost_len 10

```
`--dataset` is the dataset which you want to observe, it belongs to [`zara01`, `zara02`, `univ`, `eth`, `hotel` ].
`--ghost_len` is the length of trajectory to display, i.e. length of trajectory tail.

If you want to observe the predictions, you should rewrite the defined interface `predict`, which
takes `dataset, anno_data, frameId` as inputs and output predictions at current frame.
The outputs should be an array with `N x 3`, where each row represents 'trackid, pos_x and pos'.

The you should open the flag 'pred'
```buildoutcfg
python Marauders_Map.py --dataset zara01 --ghost_len 10 --pred True
```

