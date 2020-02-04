# TEF(TensorflowExtendFramework)

## 概述
Tensorflow是目前使用最广泛的深度学习解决方案，但是在高维稀疏数据场景（如广告、推荐、搜索等）下Tensorflow有很多不足之处：

* 参数必须以固定维度的矩阵形式提前分配（训练开始前），不支持参数的实时（训练过程中）分配与淘汰
* 不支持参数的增量形式的导出

TEF通过Operation扩展的机制，将Tensorflow的参数分配与更新任务交给自定义的参数服务器来承担，从而克服了以上的几点不足：

* 通过TEF可以很方便的对接自定义参数服务器
* 通过TEF可以很方便的实现参数的动态分配和淘汰，以及参数增量导出
* TEF以单独的python package形式安装部署


## 使用指南
主要API介绍：

|方法或类名|功能|
|---|---|
|tef.ops.variable|分配稠密参数|
|tef.ops.embedding|获取离散参数embedding|
|tef.ops.embedding_sprase|获取离散参数embedding|
|tef.training.GradientDescentOptimizer|训练优化器|


参考样例：examples/deepctr/deepctr.py


## 对接自定义参数服务器

参考样例：tef/core/kernels/ps\_client/ps\_client\_dummy.h

