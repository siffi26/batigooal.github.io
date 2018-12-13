---
title: face_papers_read.md
date: 2018-12-11 14:26:49
tags:
---
本篇博客记录一些人脸识别领域的经典论文以及最新论文的阅读笔记。

---
LFW、YTF、MegaFace测试基准

Models | LFW | YTF | MegaFace | Model Size | Training Images
---------|----------|----------|----------|----------|----------
MobiFace | 99.7% | / | 91.3% | 9.3MB | 3.8M
MobileFaceNet | 99.48% | / | 90.71% | 4MB | 3.8M
Google-FaceNet | 99.63% | / | 86.47% | 30MB | 200M
MobileNet_V1 | 99.5% | / | 92.65% | 112MB | 3.8M
CosFace | 99.73% | / | / | / | 5M
LightCNN | 99.33% | / | / | 50MB | 4M

---


#### **DeepFace**: *Closing the Gap to Human-Level Performance in Face Verification (CVPR 2014, Facebook AI)*
+ paper: [DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)

#### **DeepID**: *Deep Learning Face Representation from Predicting 10,000 Classes (CVPR 2014)*
+ paper: [DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)
+ codes: [DeepID_FaceClassify](https://github.com/stdcoutzyx/DeepID_FaceClassify)

#### **DeepID2**: *Deep Learning Face Representation by Joint Identification-Verification*
+ paper: [DeepID2](http://papers.nips.cc/paper/5416-analog-memories-in-a-balanced-rate-based-network-of-e-i-neurons)
+ codes: [基于Caffe的DeepID2实现](https://www.miaoerduo.com/deep-learning)

#### **DeepID2+**: *Deeply learned face representations are sparse, selective, and robust*
+ paper: [DeepID2+](http://arxiv.org/abs/1412.1265)
+ video: [xiaogang wang的视频讲解](http://research.microsoft.com/apps/video/?id=260023)

#### **DeepID3**: *Face Recognition with Very Deep Neural Networks (2015.2)*
+ paper: [DeepID3](https://arxiv.org/pdf/1502.00873v1.pdf)

#### **VGGFace**: *Deep Face Recognition (Parkhi15)*
+ paper: [VGGFace](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)

#### **Light CNN**: *A Light CNN for Deep Face Representation with Noisy Labels*
+ paper: [Light CNN](https://arxiv.org/pdf/1511.02683v4.pdf)

#### **FaceNet**: *A Unified Embedding for Face Recognition and Clustering (CVPR 2015, Google Inc)*
+ paper: [FaceNet](http://arxiv.org/abs/1503.03832)
+ codes: [facenet_tensorflow](https://github.com/davidsandberg/facenet), [facenet_triplet_caffe](https://github.com/hizhangp/triplet)

#### **Center Loss**: *A Discriminative Feature Learning Approach for Deep Face Recognition (ECCV 16)*
+ paper: [Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)

#### **Range Loss**: *Range Loss for Deep Face Recognition with Long-Tailed Training Data (2016.11)*
+ paper: [Range Loss](https://arxiv.org/pdf/1611.08976.pdf)

#### **Large-Margin Softmax Loss**: *Large-Margin Softmax Loss for Convolutional Neural Networks (2016.12)*
+ paper: [Large-Margin Softmax Loss](https://arxiv.org/pdf/1612.02295.pdf)
+ codes: [Large-Margin Softmax Loss](https://github.com/wy1iu/LargeMargin_Softmax_Loss)

#### **NormFace**: *NormFace: L2 Hypersphere Embedding for Face Verification (2017.4)*
+ paper: [NormFace](https://arxiv.org/pdf/1704.06369.pdf)
+ codes: [NormFace](https://github.com/happynear/NormFace)

#### **Sphereface**: *Deep hypersphere embedding for face recognition (2017.4)*
+ paper: [SphereFace](https://arxiv.org/pdf/1704.08063.pdf)

#### **Marginal Loss**: *Marginal Loss for Deep Face Recognition (CVPR 2017)*
+ paper: [Marginal Loss](https://ibug.doc.ic.ac.uk/media/uploads/documents/deng_marginal_loss_for_cvpr_2017_paper.pdf)

#### **DCFL**: *Deep Correlation Feature Learning for Face Verification in the Wild (2017.12)*
+ paper: [DCFL](http://bhchen.cn/paper/spl2017.pdf)

#### **COCO Loss**: *Rethinking Feature Discrimination and Polymerization for Large-scale Recognition (2017.10)*
+ paper: [COCO Loss](https://arxiv.org/pdf/1710.00870.pdf)
+ codes: [COCO Loss](https://github.com/sciencefans/coco_loss)

#### **AM-Softmax**: *Additive Margin Softmax for Face Verification (2018.1)*
+ paper: [AM-Softmax](https://arxiv.org/pdf/1801.05599.pdf)
+ codes: [AM-Softmax](https://github.com/happynear/AMSoftmax)

#### **CCL**: *Face Recognition via Centralized Coordinate
Learning (2018.1)*
+ paper: [CCL](https://arxiv.org/pdf/1801.05678.pdf)

#### **ArcFace**: *Additive Angular Margin Loss for Deep Face Recognition (2018.1, InsightFace)*
+ paper: [ArcFace](https://arxiv.org/pdf/1801.07698.pdf)

#### **CosFace**: *Large Margin Cosine Loss for Deep Face Recognition (2018.1, Tencent)*
+ paper: [CosFace](https://arxiv.org/pdf/1801.09414.pdf)

#### **MobileFaceNets**: *Efficient CNNs for Accurate RealTime Face Verification on Mobile Devices (2018.4)*
+ paper: [MobileFaceNets](https://arxiv.org/pdf/1804.07573v4.pdf)
+ codes: [MobileFaceNets](https://github.com/moli232777144/mobilefacenet-mxnet)

#### *Minimum Margin Loss for Deep Face Recognition*
+ paper: [Minimum Margin Loss](https://arxiv.org/pdf/1805.06741.pdf)

#### **Git Loss**: *Git Loss for Deep Face Recognition (2018.7)*
+ paper: [Git Loss](https://arxiv.org/pdf/1807.08512.pdf)
+ codes: [Git Loss](https://github.com/kjanjua26/Git-Loss-For-Deep-Face-Recognition)

#### **GridFace**: * Face Rectification via Learning Local Homography Transformations (Face++, Megvii Inc)*
+ paper: [GridFace](https://arxiv.org/pdf/1808.06210v1.pdf)

#### **Contrastive CNN**: *Face Recognition with Contrastive Convolution*
+ paper: [Contrastive CNN](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chunrui_Han_Face_Recognition_with_ECCV_2018_paper.pdf)


#### **GhostVLAD**: *GhostVLAD for set-based face recognition (2018.11, DeepMind)*
+ paper: [GhostVLAD](https://arxiv.org/pdf/1810.09951.pdf)

#### *Data-specific Adaptive Threshold for Face Recognition and Authentication (2018.11)*
+ paper: [Adaptive Threshold](https://arxiv.org/pdf/1810.11160.pdf)

---
#### **PRN**: *Pairwise Relational Networks for Face Recognition (ECCV 2018)*
+ paper: [PRN](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kang_Pairwise_Relational_Networks_ECCV_2018_paper.pdf)

本文提出一种新颖的人脸识别方法，叫做pairwise relational network(PRN)，它可以围绕特征图的关键点获取局部外观patches，在成对的局部外观patches之间捕获pairwise relation。PRN可以捕获到不同ID之间独特的判别性的成对关系。文章添加一个人脸ID状态特征，这是通过在特征图上的序列化局部外观patches的方式，从LSTM单元网络中获取。为了进一步提升人脸识别的准确性，文章将全局外观表达和成对关系特征相融合。

文章的主要贡献包括：
+ 提出一种新的使用pairwise relational network(PRN)的人脸识别方法，PRN在特征图的局部外观patches上捕获到独一无二的具有判别性的pairwise relational，用于在不同ID中区分人脸图片
+ 所提出的PRN无论是对1:1还是1:N的准确率都有所增加
+ 在LFW，YTF，IJB-A，IJB-B几个公共数据集上大部分都是SOT的水平

文章所提出的方法在细节上包括：用于global appearance representation的骨干网络、人脸对齐、pairwise relational network、带有face identity states的pairwise relational network，损失函数五个部分：

**基本的CNN骨干网络**：使用一些3-layer的Residual Bottleneck Blocks组成骨干网络。输入140×140的人脸图片，网络结构如下：

Layer name | Output size | 101-layer
---------|----------|----------
 conv1 | 140×140 | 5×5, 64
 conv2_x | 70×70 | 3×3 max pool, /2
 conv2_x | 35×35 | $\begin{bmatrix} 1×1, 64 \\ 3×3, 64 \\ 1×1, 256 \end{bmatrix}$×3
 conv3_x | 35×35 | $\begin{bmatrix} 1×1, 128 \\ 3×3, 128 \\ 1×1, 512 \end{bmatrix}$×4
 conv4_x | 18×18 | $\begin{bmatrix} 1×1, 256 \\ 3×3, 256 \\ 1×1, 1024 \end{bmatrix}$×23
 conv5_x | 9×9 | $\begin{bmatrix} 1×1, 512 \\ 3×3, 512 \\ 1×1, 2048 \end{bmatrix}$×3
 - | 1×1 | global average pooling, 8630-d fc, softmax

global appearance representation（全局外观表达）:GAP全局平均池化输出的2048维向量
local appearance representation（局部外观表达）:在conv5_3层的特征图9×9×2048上根据人脸关键点进行ROI投影

**人脸对齐**：


---
#### **MobiFace**: *A Lightweight Deep Learning Face Recognition on Mobile Devices (2018.11)*
+ paper: [MobiFace](https://arxiv.org/pdf/1811.11080)

本文提出一种新的DNN叫做MobiFace，简单但是高效，适合在移动设备用于人脸识别。LFW上可以达到99.7%，Megaface上可以达到91.3%。文章引入一种新颖的**lightweight**并且**highperformance**的DNN用于移动设备上的人脸识别，文章的贡献包括：
1. 对成功的MobileNet框架进行了改进提升，更轻量化权重并且更好的网络MobiNet模型适合部署在移动设备上；
2. 所提出的MobiNet适合用于人脸识别，是可以端到端优化的深度学习框架；
3. 在LFW和Megaface两个数据集上，与其他基于移动端的网络和大规模深度网络在人脸识别任务上进行了对比。

与MobiNet的网络比较接近的两种轻量级设计包括：
1. 紧凑型模块的设计。**layers can abate the number of weights, help use less memory, and mitigate heavy computation cost for inference stage.** MobileNet中提出使用depthwise separable convolution替代标准的convolution，可以减少大量参数。MobileNet的参数数量是4.2M，加乘运算数量569M，在ImageNet的分类数据集上达到70.6%的准确率；对比来看，VGG-16参数数量138M，加乘运算数量15300M，准确率仅71.5%。MobileNet-V2版本提出inverted residuals和linear bottlenecks，参数数量3.4M，加乘运算数量300M，准确率72%。另一方面，depthwise convolution在Caffe，Pytorch，Tensorflow等框架上并不能高效的使用CPU运算。MobileFaceNet使用global depthwise convolution层替换global averagingpooling层，对不同位置的像素进行不同的加权。

2. 网络剪枝。由于每个已修剪的连接，需要将索引列表存储在内存中，从而导致训练和推理的速度非常慢。

MobiNet的介绍：
**网络设计策略**
1. **Bottleneck Residual block with the expansion layers**：

2. **Fast Downsampling**：紧凑的网络需要让输入图片到输出图片的信息传递最大化，从而避免特征图大空间维度上的高计算成本。文章认为大规模的深度网络通常都是采用非常缓慢的downsampling，这样做的目的是为了保留住更多的细节信息。但是资源受限的情况下，缓慢的downsampling会带来两个问题，一个是保留了不重要的特征，另一个是耗时。因此，文章提出采用快速downsampling策略。 

**MobiFace**
用于人脸识别的MobiFace，给定输入人脸图片112×112×3，网络结构如下图：

Input | Operator
---------|----------
 112×112×3 | 3×3 Conv, /2, 64
 56×56×64 | 3×3 DWconv, 64
 56×56×64 | Block 1×$\begin{cases} 1×1 Conv, 128\\ 3×3 DWconv, /2, 128\\ 1×1 Conv, Linear, 64 \end{cases}$
 28×28×64 | RBlock 2×$\begin{cases} 1×1 Conv, 128\\ 3×3 DWconv, 128\\ 1×1 Conv, Linear, 64 \end{cases}$
 28×28×64 | Block 1×$\begin{cases} 1×1 Conv, 256\\ 3×3 DWconv, /2, 256\\ 1×1 Conv, Linear, 128 \end{cases}$
 14×14×128 | RBlock 3×$\begin{cases} 1×1 Conv, 256\\ 3×3 DWconv, 256\\ 1×1 Conv, Linear, 128 \end{cases}$
 14×14×128 | Block 1×$\begin{cases} 1×1 Conv, 512\\ 3×3 DWconv, /2, 512\\ 1×1 Conv, Linear, 256 \end{cases}$
 7×7×256 | RBlock 6×$\begin{cases} 1×1 Conv, 512\\ 3×3 DWconv, 512\\ 1×1 Conv, Linear, 256 \end{cases}$
 7×7×256 | 1×1 Conv, 512
 7×7×512 | 512-d FC

其中，Block即Bottleneck block，RBlock即Residual Bottleneck block，在每个convolution层之后都会使用BN层和非线性激活PReLU，线性convolution层除外。最后一层，使用全连接层取代其他文献经常使用的全局平均池化，因为全连接层公平的对特征图的每个单元进行加权处理。

实验结果
使用干净的MS-Celeb-1M作为训练集，85K个ID，3.8M张图片，使用LFW和Megaface作为测试集。

MS-Celeb-1M数据集清理的方法：计算每个ID的center feature，使用到ID center的距离来对他们的人脸图片进行排序，远离center的自动被删除，同时配合人工检测。

LFW的标注人脸：5749个ID，13233张图片，使用测试工具处理后包括6000对人脸，其中有一半是正样本对。

MegaFace：包括两个主要的数据集，gallery中包括690K个ID，超过1M的图片，probe中有两个子集，probe子集包括530个ID的100K张图片，FGNET子集包括0-69岁的82个ID的1002张图片。

实现细节
人脸检测以及五个关键点的检测都使用MTCNN，对齐后图片尺寸为112×112×3，归一化为[-1,1]。batch size设为1024，使用随机梯度下降（SGD）优化，momentum设为0.9，初始学习率为0.1，在40K，60K，80K迭代时按10倍的倍率下降，训练在100K迭代时截止。


