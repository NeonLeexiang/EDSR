# EDSR - Pytorch 

同样是 `Super Resolution` 领域的一个经典文章，有了 `SRCNN` 的一个基础, 以及我们上次复现了 `VDSR` 还有 `SRGAN` 
这次的论文复现我们选择复现 `EDSR` 它和 `SRGAN` 有着类似的 `ResBlock` 结构，只不过不同的是通过研究发现 `BatchNormal` 
虽说对训练有着非常高的速度上面的提高，但是对结果的影响甚微，所以 `EDSR` 的 `ResBlock` 相对于 `SRGAN` 来说有着小小的区别。
它去除了 `SRGAN` 中表现非常不错的 `BatchNormal`, 同样的，也在模型的 `Loss` 上用了 `L1Loss` 而不是其他的 `MSELoss` 之类的。
这一次我们直接把它搭建成我们比较常用的 `torch` 的方式。  
<br> 这一次的 `torch` 复现利用以前的复现内容一下子就搭建起来了，后续也会慢慢的优化和review原来的代码。


## EDSR 论文重点
`EDSR` -> [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)

作者提出的模型主要是提高了图像超分辨的效果，并赢得了NTIRE2017超分辨率重建挑战赛。
做出的修改主要是在ResNet上。
作者移除了残差结构中一些不必要的模块如BN层，结果证明这样确实有效果。
另外，作者还提出了一种多尺度模型，不同的尺度下有绝大部分参数都是共享的。
这样的模型在处理每一个单尺度超分辨下都能有很好的效果。

#### EDSR 的特别之处
 * `EDSR` 最有意义的模型性能提升是去除掉了 `SRResNet` 的批量标准化 (batch normalization, BN) 层。   
     

    由于批量标准化层对特征进行了规范化，因此通过规范化特征可以摆脱网络的范围可变性，最好将其删除，从而可以扩大模型的尺寸来提升结果质量。
    此外，由于 BN 层消耗的内存量与前面的卷积层相同，因此去掉BN层后，`EDSR` 的 GPU 内存使用量也会减少。
    与 `SRResNet` 相比，由于 BN 层的计算量和一个卷积层几乎持平，baseline 模型没有批量标准化层，在训练期间可节省大约40％的内存使用量。
    因此，可以在有限的计算资源下构建一个比传统 ResNet 结构具有更好性能的更大模型。
   
 * `EDSR` 使用的损失函数是 `L1Loss`  
     

    训练时，损失函数用 L1 而不是 L2 ，即根据 `LapSRN` 的思想采用了 L1 范数来计算对应的误差，L2 损失会导致模糊的预测。
    BN 有一定的正则化效果，可以不去理会 Dropout ，L2 正则项参数的选择。
    除此之外，更深层的原因是是实际图像可能含有多种特征，对应有关的图像构成的真实分布。
    图像特征分布有许多个峰值，比如特征1是一个峰，特征2是一个峰...
    对于这种图像分布，我们称之为：多模态 (Multimodal) 。
    假如用 MSE（或者 L2 ）作为损失函数，其潜在的假设是我们采集到的样本是都来在同一个高斯分布。
    但是生活中的实际图像具有多种特征，而且大部分图像分布都不只有一个峰。
    如果强行用一个单峰的高斯分布，去拟合一个多模态的数据分布，例如两个峰值。
    因为损失函数需要减小生成分布和数据集经验分布（双峰分布）直接的差距，而生成分布具有两种类型，模型会尽力去“满足”这两个子分布，最后得到的优化结果。
    
    简而言之，当我们在使用L2损失训练出来的分布中采样时，虽然采集到的样本属于数据集真实分布的几率很低，但是由于处于中间位置，会被大量采集出来。 
    故我们最终得到的生成样本实质上是多种特征的数据样本特征信息的平均效果，故产生模糊图像。 
    也就是生成的图像中的某些信息很可能不属于特征所要表现的任何一个

 * `EDSR` 的残差缩放 `Residual Scaling`    
  
  
    EDSR 的作者认为提高网络模型性能的最简单方法是增加参数数量，堆叠的方式是在卷积神经网络中，堆叠多个层或通过增加滤波器的数量。
    当考虑有限的复合资源时，增加宽度 (特征Channels的数量) F 而不是深度(层数) B 来最大化模型容量。
    但是特征图的数量增加(太多的残差块)到一定水平以上会使训练过程在数值上不稳定。
    残差缩放 (residual scaling) 即残差块在相加前，经过卷积处理的一路乘以一个小数 (作者用了0.1)。
    在每个残差块中，在最后的卷积层之后放置恒定的缩放层。
    当使用大量滤波器时，这些模块极大地稳定了训练过程。
    在测试阶段，该层可以集成到之前的卷积层中，以提高计算效率。
    使用上面三种网络对比图中提出的残差块（即结构类似于 SRResNet ，但模型在残差块之外没有 ReLU** 层）构建单尺度模型 EDSR。
    此外，因为每个卷积层仅使用 64 个特征图，所以单尺度模型没有残差缩放层。


#### BatchNormal layer 的介绍
 * `BatchNormal` 的介绍  
   

    Batch Norm可谓深度学习中非常重要的技术，不仅可以使训练更深的网络变容易，加速收敛，还有一定正则化的效果，可以防止模型过拟合。在很多基于CNN的分类任务中，被大量使用。
    但在图像超分辨率和图像生成方面，BatchNorm的表现并不是很好。当这些任务中，网络加入了BatchNorm层，反而使得训练速度缓慢且不稳定，甚至最后结果发散。


 * Super Resolution 上使用 `BatchNormal` 不好的原因
  
 
    以图像超分辨率来说，网络输出的图像在色彩、对比度、亮度上要求和输入一致，改变的仅仅是分辨率和一些细节。
    而Batch Norm，对图像来说类似于一种对比度的拉伸，任何图像经过Batch Norm后，其色彩的分布都会被归一化。
    也就是说，它破坏了图像原本的对比度信息，所以Batch Norm的加入反而影响了网络输出的质量。
    ResNet可以用BN，但也仅仅是在残差块当中使用。
    还是回到SRResNet，上图的(b)就是一个用于图像超分辨率的残差网络。


 * `SRGAN` 上用 `BatchNormal` 的原因


    ResNet中引入了一种叫残差网络结构，其和普通的CNN的区别在于从输入源直接向输出源多连接了一条传递线来恒等映射，用来进行残差计算。
    可以把这种连接方式叫做identity shortcut connection,或者我们也可以称其为skip connection。
    其效果是为了防止网络层数增加而导致的梯度弥散问题与退化问题。


 * `BatchNormal` 在分类问题上面 
   

    图像分类不需要保留图像的对比度信息，利用图像的结构信息就可以完成分类。
    所以，将图像信息都通过BatchNorm进行归一化，反而降低训练难度。
    甚至，一些不明显的结构，在BatchNorm后也会被凸显出来（对比度被拉开）。

 * `BatchNormal` 简而言之


    BN会是网络训练时使数据包含忽略图像像素（或者特征）之间的绝对差异（因为均值归零，方差归一），而只存在相对差异。
    所以在不需要绝对差异的任务中（比如分类），BN提升效果。
    而对于图像超分辨率这种需要利用绝对差异的任务，BN会适得其反。


## 以下是对应的之前 Super Resolution 的论文重点：

#### SRGAN 论文重点
在实现 `SRGAN` 论文之前我们实现了我们的传统的 `VDSR` 以及更加前面我们实现过的 `SRCNN` 这两个都是比较入门的 `Super Resolution` 方向的论文。    
`GAN` 神经网络是一个非常特别的神经网络，训练会极其复杂。总体来说 `SRGAN` 的 `generator` 使用的是 `VGG19`。

#### VDSR
`VDSR`  [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf)  
以及 `VDSR` 论文的重点为：
* 模型具有非常深的层
* 使用了残差学习和自适应梯度裁剪来加速模型的训练
* 将单一倍数的超分模型扩充到多个倍数  
  
与`SRCNN`一样，都是先将低分辨率输入双三次插值到高分辨率，再来进行模型的预测。
这里包括两个部分，VGG-like的深层网络模型，每一层卷积中均使用带padding的3x3卷积层，
并且随后都会添加一个ReLU来增强模型的非线性，这里与`SRCNN`、`SRCNN-Ex`都有着较大的改变。
然后最后使用残差学习来将模型预测到的结果element-wise的形式相加，来得到最终的结果。  

#### SRCNN
`SRCNN` [Image super-resolution using deep convolutional networks](https://ieeexplore.ieee.org/document/7115171/;jsessionid=sqmfzoJEerWjinbTLnm8TVyWaFJSTAXKVbNp_abvj-XrT4nB9Sf6!84601464)
`SRCNN` 是 `end-to-end`（端到端）的超分算法，所以在实际应用中不需要任何人工干预或者多阶段的计算.

#### SRGAN
`SRGAN` [" Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)  
运用了 `SR Res Net` 的概念，运用了残差



## Datasets
之前我们的 `SRCNN` 使用的是 `cifar-10` , 那个数据集不是很好用，因为它的原始数据就是 `32 * 32` 大小的，
不太适合放大缩小。我们尝试试试其他的数据集看看效果会是怎样。  
然后我们选用一个比较大的数据集也就是我们的 `BSD500` 的数据集。
数据集选用的是 `BSD500` 但是我们截取成为 `128*128`。  


## Prerequisites
 * pytorch > 1.0


## Usage
~~ignore the usage~~
For training, `python trains.py`
<br>
For testing, `python trains.py` 暂时没有写对应的 `test.py`


## EDSR Problems:
可能是因为 `MeanShift` 的原因，图像的像素转化有些问题，导致 `pnsr` 值非常低同时整个图像像素也很奇怪。
同时对 `MeanShift` 的了解不够多，一开始的 `rgb range` 设定为255，其实没有注意到 `ToTensor()` 
的时候，函数已经帮转化成0到1之间的浮点类型数了。


[comment]: <> (## Problems  )

[comment]: <> (目前因为分配的显存没有办法做到大小为 `256*256` 的超分辨，所以 `torch.cuda&#40;&#41;` 会超出显存大小，)

[comment]: <> (如果我们将整个网络的大小缩放放小，我们的训练将会更加快速。    )
  
[comment]: <> (有一个想法就是我们缩放我们的 `crop_size=128` 同时因为我们是喂入一个 `batch` 的数据进入 `cuda` ， )

[comment]: <> (我们可以设置我们的 `batch_size` 更小，虽说一个 `epoch` 会训练更多速度，但是相应的训练速度也会更快。)

[comment]: <> (同时我们也需要考虑我们的数据过拟合的情况。)

[comment]: <> (* 因为使用的是 `cifar10`的数据集，会出现的问题就是它的图像数据的大小是 `32*32` 的，)

[comment]: <> (  所以没有做一些放大缩小的操作获取对应的 High Resolution Image -> Low Resolution Image 的操作。)
  
[comment]: <> (* 做的 `Keras` 和 `Tensorflow` 的训练并没有像 `Pytorch` 一样使用 `tqdm` 模块去做一些操作。  )
  
[comment]: <> (* `pytorch` 要非常注意一点就是它的 Tensor 和 `tensorflow` 或者 `keras` 不一样，可能 `tensorflow` `keras` 是以)

[comment]: <> (  `Size * H * W * C` 而 `pytorch` 是以 `Size * C * H * W` 的方式去计算的，所以使用的数据需要通过 `torch.permute` 的 方式修改数据格式。)
  
[comment]: <> (* `pytorch` 的复现有许多代码上的不理解，后续慢慢解决。)
  

## Result

以下是 `EDSR` 的 `result table`:

| Dataset | Epochs | Module | Method | psnr  | 
| ------- | ------ | ------ | ------ | ----- |
| BSD500  |  200   | EDSR   | pytorch| 23.38 |
| BSD500  |  400   | EDSR   | pytorch| 23.18 |


训练了 200 个 Epochs 的 `EDSR`:
  

| Bicubic | High Resolution | Super Resolution |
|---------|---------------- |----------------- | 
![avatar](edsr_torch_model_file/training_results/SRF_4/epoch_200_index_5.png)
![avatar](edsr_torch_model_file/training_results/SRF_4/epoch_200_index_16.png)
  


训练了 400 个 Epochs 的 `EDSR`:
  

| Bicubic | High Resolution | Super Resolution |
|---------|---------------- |----------------- | 
![avatar](edsr_torch_model_file/statistics/epoch_400_index_5.png)
![avatar](edsr_torch_model_file/statistics/epoch_400_index_16.png)
  



  
以下是 `VDSR` 的 `result table` :  

| Dataset | Epochs | Module | Method     | psnr   |
|---------|------- |------  |------      | ------ |
| cifar10 | 500    | SRCNN  | tensorflow | 56.0   |
| cifar10 | 500    | SRCNN  | keras      | 25.9   |
| cifar10 | 500    | SRCNN  | pytorch    | 26.49  |


以下是 `SRGAN` 的 `result table`: 

| Dataset | Epochs | Module | Method | psnr | 
| ------- | ------ | ------ | ------ | ---- |
| BSD500  |  200   | SRGAN  | pytorch| 22.4 |
| BSD500  |  400   | SRGAN  | pytorch| 22.6 |



[comment]: <> (训练了 200 个 Epochs 的 `SRGAN` ：)
  
[comment]: <> (分别为)

[comment]: <> (| Bicubic | High Resolution | Super Resolution |)

[comment]: <> (|---------|---------------- |----------------- | )

[comment]: <> (![avatar]&#40;srgan_torch_model_file/training_results/SRF_4/epoch_200_index_1.png&#41;)

[comment]: <> (![avatar]&#40;srgan_torch_model_file/training_results/SRF_4/epoch_200_index_6.png&#41;)
  
[comment]: <> (训练了 400 个 Epochs 的 `SRGAN` ：)

[comment]: <> (![avatar]&#40;srgan_torch_model_file/training_results/SRF_4/epoch_400_index_2.png&#41;)

[comment]: <> (![avatar]&#40;srgan_torch_model_file/training_results/SRF_4/epoch_400_index_5.png&#41;)

[comment]: <> (<img src="srgan_torch_model_file/training_results/SRF_4/epoch_200_index_1.png" alt="Epochs 200">)

  
[comment]: <> (`tensorflow` 可能是因为数据集的问题导致 `psnr` 的计算会出现一些小的问题)

[comment]: <> (因为数据集的使用问题，所以模型的训练是没有意义的。  )

[comment]: <> (出于对`cifar`数据集的一个不了解，它是 `32*32`的，但是我将它 bicubic 放大成了 `128*128` 作为 ground true。  )

[comment]: <> (然后训练数据 从 `32*32` resize 到 `32*32` 用邻近插值，然后又 bicubic 放大成 `128*128` 作为训练数据，这个是无效的训练。)

[comment]: <> (所以训练效果直接爆炸。  )

[comment]: <> (后续也不因数据集问题做更多的尝试和改进。整个内容当作对 `tensorflow > 2.0`  的一个入门尝试。)

## References
`EDSR` -> [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)

```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```
  

A PyTorch implementation of `EDSR` based on CVPR 2017 paper
  
This repository is implementation of the [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)
And code [sanghyun-son/EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch)


## More Datasets

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
| 91-image | 4 | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0) |


