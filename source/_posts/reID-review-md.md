---
title: 行人再识别研究综述
date: 2018-07-02 19:21:44
tags:
---

## Re-ID基本概念
定义：跨摄像头跨场景下行人的识别与检索。

技术难点：可能存在无正脸的照片、姿态、配饰、遮挡、相机拍摄角度差异、图片模糊、环境变化、服装更换、跨季节、光线差异等。

常用数据集：
+ Market1501：6个摄像头（5个高分辨率，1个低分辨率），视野范围存在重叠，包括32668个行人目标框，包括了1501个人。每个人至少出现在两个摄像头中，存在2793个干扰项。Market1501+500K用于测试模型是否过拟合Market1501。
+ DukeMTMC-reID：8个摄像头，1080p图片，36411张图，1812个人。
+ CUHK03：10个摄像头，1467个人，总共13164个行人目标框。

评价指标：
+ Rank-1：首位命中率，存在偶然因素
+ mAP：平均精度均值，先单独计算每张测试图片的精度，然后计算均值

## Re-ID相关文章介绍
### CVPR2018中的Re-ID
#### 1. Pose Transferrable Person Re-Identification
摘要：提出一个可迁移的Re-ID框架，利用迁移后的样本扩充（即ID监督）来增强Re-ID模型的训练。

网络结构：
![](cut-imgs/2018-07-02-20-18-30.png)

效果：
Market-1501的mAP是57.98，rank-1是79.75
DukeMTMC-reID的mAP是48.06，rank-1是68.64
CUHK03(labeled)的mAP是30.5，rank-1是33.8
CUHK03(detected)的mAP是28.2，rank-1是30.1

#### 2. Deep Spatial Feature Reconstruction for Partial Person Re-identification: Alignment-free Approach
摘要：解决有遮挡的情况下的行人重识别问题。使用图像空域重建的方法得到与输入图像尺寸一致的空域特征图，借鉴字典学习中重建误差来计算不同的空域特征图的相似度。