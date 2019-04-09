# HAT

Hust Artificial intelligence Trainer.

这是一个运行神经网络的框架，使神经网络的构建、训练、测试、部署等变得更简单轻松。

* 数据集和模型模块化，可自由增删
* 通过输入指令指定数据集、模型、参数等，也可以指定模式

## 环境

以下为本人的环境

* Windows 10 专业版 1809
* Python: 3.6.8
* Anaconda: 4.6.8
* Tensorflow-gpu: 2.0 alpha
* CUDA: 10.0.130
* cuCNN: 7.5.0

**注：tensorflow 1.13可能也可以使用**

## 部署

下载本项目文件，放在任意文件夹内即可

## 开始使用

* 在IDE里面选择`main.py`文件，并`运行`/`调试`
* 直接运行
```
python main.py
```

打印了一些tensorflow的信息之后，会出现一个输入提示符

```
=>
```

框架本身有默认值，所以第一次运行可以直接输入回车运行，会有日志打印出来，可以看到使用的数据集、模型、以及其他信息

```
################################
[logs] 2019-04-06-01-08-10
[logs] Datasets: mnist
[logs] Models: MLP
[logs] Epochs: 5
[logs] Batch_size: 128
[logs] Using Optimizer: adam
[logs] h5 not exist, create one
[logs] logs dir: logs\mnist_MLP\mnist_MLP_0\
```

第二次运行，再次输入回车，日志如下，读取了上一次训练保存的模型
```
[logs] h5 exist: logs\mnist_MLP\mnist_MLP_0.h5
```

## 进阶

运行主程序的时候，可以输入自定义的参数，如下
```
=>datasets=cifar10 model=LeNet batch_size=128 epochs=150
```

另外，还可以选择模式
```
=>test-only
```

以下列出部分数据集、模型，所有可输入参数和模式，部分支持简写

>数据集：datasets(dataset, data, dat)
>>mnist<br>
cifar10<br>
fashion_mnist

>模型：models(model, mod)
>>MLP<br>
>>LeNet<br>
>>VGG16<br>
>>GoogleNetV2

>参数
>>batch_size(bat)<br>
>>epochs(epoch, ep)

>模式
>>训练：train-only(train-o, train)<br>
>>测试：test-only(test-o, test)<br>
>>生成图像：gimage(gimg)

## 创建模型 已过时!!!等待修改

创建自己的模型非常简单，只需要创建一个文件，放在models文件夹。该文件夹内有一些模型，可供使用和参考

注：该项目使用keras创建模型，在models内有三种创建模型的方法可供参考，分别是`MLP.py`, `LeNet.py`, `GoogleNetV2.py`

### 以MLP模型为例

```
'''
  默认模型
  简单三层神经网络[添加了Dropout层]
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           394,634
    Trainable params:       394,634
    Non-trainable params:   0
'''


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *


################################################################
# MBP model args

MBP_LOCAL_SIZE = 128
MBP_DROP_RATE = 0.5

################################################################

def MLP(i_s, i_d, n_s):
  model = Sequential([
    Flatten(input_shape=(i_s, i_s, i_d)),
    Dense(MBP_LOCAL_SIZE, activation='relu'),
    Dropout(MBP_DROP_RATE),
    Dense(n_s, activation='softmax')
  ])
  return model

# test part
if __name__ == "__main__":
  mod=MLP(32, 3, 10)
  print(mod.summary())

```

### 解析

**文件文档（可选）** 描述模型，并不是必须的。

**引包** 视情况而定，根据不同构造方法以及不同的需求自行引包

**模型参数** 没有必须的参数，注意格式规范即可。注意前缀，比如模型名为MLP，参数的前缀也最好为MLP。**例外**：当这个变量并不希望被模型外访问时，在变量前加上`_`，如`_COUNT_CONV`。另外，部分参数可能会被检测，比如OPT（自定义优化器），命名不当将导致检测不到相关参数。

**其他函数（可选）** 在第三种构建模型的方法里可能需要用到自定义函数，详细可参考`GoogleNetV2`的用法

**主函数（必须）** 注意：
* 函数名跟文件名一致
* 变量前缀跟函数名一致
* 注意函数的参数，i_s: image_size, i_d: image_depth, n_s: num_classes。**注：暂时还未考虑非正方形图像，故i_s为整数，后续会修改相关接口，改成shape-3D输入**
* 返回模型

**测试部分（可选）** 在本例中测试部分中，用`cifar10`数据集图像尺寸构建了一个模型，并且调用了`summary()`函数来打印模型的一些层的情况。**注：为了不影响主程序调用，建议跟本例一样写在`if __name__ == "__main__":`里**

### 入库

文件保存好之后，主函数并不能直接调用模型，需要入库。打开`__init__.py`文件，参照其他模型，添加一句代码即可。
```
from .MLP import *
```
注：`.`不能漏写，相对引包的标志

## 创建数据集 已过时!!!等待修改

>使用Tensorflow官方的Datasets创建

### 以mnist数据集为例

```
from .Packer import Packer
import tensorflow.keras.datasets as ds


################################################################
# mnist datasets args

MNIST_NUM_TRAIN = 60000
MNIST_NUM_TEST = 10000
MNIST_NUM_CLASSES = 10
MNIST_IMAGE_SIZE = 28
MNIST_IMAGE_DEPTH = 1

################################################################

def mnist():
    mnist = Packer()
    (mnist.train_images, mnist.train_labels), (mnist.test_images, mnist.test_labels) = ds.mnist.load_data()
    mnist.train_images, mnist.test_images = mnist.train_images / 255.0, mnist.test_images / 255.0
    mnist.train_images = mnist.train_images.reshape((MNIST_NUM_TRAIN, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_IMAGE_DEPTH))
    mnist.test_images = mnist.test_images.reshape((MNIST_NUM_TEST, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_IMAGE_DEPTH))
    return mnist

```

### 解析

**引包** 引入`Packer`（必须），引入Tensorflow官方的datasets

**数据集参数（必须）** `NUM_CLASSES`, `IMAGE_SIZE`, `IMAGE_DEPTH`这三项是必须要有的。

**主函数（必须）** 注意：
* 数据类型是numpy数组
* 图像应该为4D，[张数, 宽度, 高度, 深度]
* 图像数字处理成 0~1
* 返回的是Packer对象

>自行创建数据集

待补充

