# TEF(TensorflowExtendFramework)

## 概述
Tensorflow是目前使用最广泛的深度学习解决方案，但是在高维稀疏数据场景（如广告、推荐、搜索等）下Tensorflow有很多不足之处：

* 参数必须以固定维度的矩阵形式提前分配（训练开始前），不支持参数的实时（训练过程中）分配与淘汰
* 不支持参数的增量形式的导出

TensorflowExtendFramework（以下简称TEF）是笔者开源的针对高维稀疏数据场景（如广告，推荐，搜索等）的深度学习解决方案.

TEF通过Operation扩展的机制，将Tensorflow的参数分配与更新任务交给自定义的参数服务器来承担，从而克服了以上的几点不足：

* 通过TEF可以很方便的对接自定义参数服务器
* 通过TEF可以很方便的实现参数的动态分配和淘汰，以及参数增量导出
* TEF以单独的Python Package形式安装部署

## 编译与安装

1.构建开发docker镜像

```
cd docker/develop/
docker build -t tef_develop .
```

2.启动docker开发环境

```
docker run -it --net=host tef_develop

```

2.编译，生成Python Package安装包

```
git clone https://github.com/jony0917/tensorflow-extend-framework.git
cd tensorflow-extend-framework
mkdir build
cd build
cmake ..
make tef
```

3.pip安装tef

```
pip install build/tef/python/dist/tf-x.x.x.x-py2-none-any.whl
```

4.运行example,确认正确安装

```
cd examples/deepctr
python deepctr.py
```

看到类似一下输入，表明安装正确：

```
...
batch=9, loss=0.23234
...
```

## 使用指南

1. 通过TEF，你可以通过以下简单两步构建自己的支持高维稀疏数据的场景的深度学习解决方案：

* 首先你需要有自己的参数服务器，或则使用第三方参数服务器，如pslite，ps_plus等
* 然后为你的参数服务器实现接口 tef/core/kernels/ps\_client.h:PsClient， 参考样例：tef/core/kernels/ps\_client/ps\_client\_dummy.h

2. 主要API介绍：

|方法或类名|功能|
|---|---|
|tef.ops.variable|分配稠密参数|
|tef.ops.embedding|获取离散参数embedding|
|tef.ops.embedding_sprase|获取离散参数embedding|
|tef.training.GradientDescentOptimizer|训练优化器|


参考样例：examples/deepctr/deepctr.py

## 设计文档

[TensorflowExtendFramework](https://blog.csdn.net/gaofeipaopaotang/article/details/104182284)
