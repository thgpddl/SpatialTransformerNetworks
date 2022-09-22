# SpatialTransformerNetworks
SpatialTransformerNetworksOnMNIST

本项目使用Pytorch教程[SPATIAL TRANSFORMER NETWORKS TUTORIAL](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html#depicting-spatial-transformer-networks)的代码，
并加以修改。主要用来通过理论和实践来学习Spatial Transformer Networks。

实验的目的是在MNIST数据集上构建一个常规卷积+全连接层的分类模型，并将Spatial Transformer Networks插入，进行MNIST分类。

理论部分见：[]()

# 1、如何使用
先安装requirements.txt文件中的库

然后直接运行main.py文件即可

# 2、注意事项
- 代码会从Internet下载MNIST数据集，所以请保持网络畅通
- 每训练一个epoch，都会调用visualize_stn将SpatialTransformerNetworks前后效果保存到visual/文件夹下
- 训练结束后，会调用loop.show()将Test Acc变化曲线保存到result.jpg中

# 3、效果展示
Spatial Transformer Networks对MNIST的“纠正”效果（epoch=20时的效果）
![image](https://user-images.githubusercontent.com/48787805/191738390-e3719912-7b62-469e-a7d8-d298914927f6.png)


Test Acc

![image](https://user-images.githubusercontent.com/48787805/191738313-15b6711b-d21e-4d31-80f8-fef993d7aee2.png)
