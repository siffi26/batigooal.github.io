---
title: Paper笔记——PyramidBox & SNIPER
date: 2018-09-29 13:31:04
tags:
---

带来两篇关于目标检测、多尺度、小目标方面近期的文章

---

PyramidBox: A Context-assisted Single Shot Face Detector
文章提出了一种新的结合上下文的单阶段人脸检测器来解决高难度人脸检测的问题。文章通过三个方面提升了上下文信息的作用：
+ 设计了一个新的上下文anchor用来监督high-level上下文特征的学习，使用一种叫做PyramidAnchors的半监督方法
+ 提出了low-level的Feature Pyramid Network将high-level上下文语义特征和low-level人脸特征进行足够的融合，这也让PyramidBox能够在single shot中预测多个尺度的人脸
+ 引入了一种上下文敏感的结构来提升预测网络的能力，从而提升最后输出的准确度。

除此之外，作者使用了data-anchor-sampling的方法来增强训练样本的不同尺度，这样可以增加小脸训练样本的多样性。

通过利用上下文的价值，PyramidBox在两个通用的人脸检测基准数据集上取得了SOT的性能。


---
SNIPER: Efficient Multi-Scale Training
文章提出了一种叫做SNIPER的算法，该算法在实例级视觉任务中可以非常有效的进行多尺度训练。取代处理一张图像金字塔里面的每个像素，SNIPER处理合适尺度下ground-truth实例周围的上下文区域。对于背景的采样，用一个短期训练所得到的RPN网络输出的推选区域来生成这些上下文区域。和COCO数据集中800×1333像素的单个尺度训练相比，SNIPER仅仅多处理30%的像素。但是它还观察到来自图像金字塔的极端分辨率样本，比如1400×2000.即使使用resnet-101骨干网络，SNIPER也可以在单个GPU上使用20的batchsize。SNIPER将实例级别的目标检测任务的训练变得更接近图像分类任务的方式，并且建议通常需要在高分辨率上训练实例级别的视觉任务这个观点可能并不正确。

