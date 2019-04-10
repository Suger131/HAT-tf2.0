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
* Visual Studio Code
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
>>GoogleNet

>参数
>>batch_size(bat)<br>
>>epochs(epoch, ep)<br>
>>mode

>模式（注：等同于mode=x，如`gimg`等同于`mode=gimg`）
>>训练：train-only(train-o, train)<br>
>>测试：test-only(test-o, test)<br>
>>生成图像：gimage(gimg)

**注意**：框架里面涉及到三种参数，一种是交互输入参数，一种是数据集/模型自带参数，一种是框架内用户默认参数（可自行修改）。参数优先级为：交互输入参数>数据集/模型自带参数>用户默认参数。

## 创建模型

创建自己的模型非常简单，只需要创建一个文件，放在models文件夹。文件夹内有一些模型，可供使用和参考。另外，本项目使用keras创建模型。

### 以MLP模型为例，第一种模型构建方法

```
'''
  默认模型
  简单三层神经网络[添加了Dropout层]
  本模型默认总参数量[参考基准：cifar10模型]：
    Total params:           394,634
    Trainable params:       394,634
    Non-trainable params:   0
'''

from .Model import BasicModel
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *


class MLP(BasicModel):

  def __init__(self, i_s, i_d, n_s, Args):
    super(MLP, self).__init__(Args)
    self.INPUT_SHAPE = i_s
    self.INPUT_DEPTH = i_d
    self.NUM_CLASSES = n_s
    self.LOCAL_SIZE = 128
    self.DROP_RATE = 0.5
    self.model = Sequential([
      Flatten(input_shape=(self.INPUT_SHAPE, 
                           self.INPUT_SHAPE, 
                           self.INPUT_DEPTH)),
      Dense(self.LOCAL_SIZE, activation='relu'),
      Dropout(self.DROP_RATE),
      Dense(self.NUM_CLASSES, activation='softmax')
    ])
    self.check_save()

# test part
if __name__ == "__main__":
  mod = MLP(32, 3, 10, None)
  print(mod.model.summary())
```

* **文件文档（可选）** 描述模型，并不是必须的。

* **引包** 引入`BasicModel`类，以及keras里的`Sequential`和`Layers`

* **构建类** 该类继承`BasicModel`类，需要注意的点有：
  * 类名和文件名最好一致，且为模型本身的名字
  * 必须具有的参数：`INPUT_SHAPE`, `INPUT_DEPTH`, `NUM_CLASSES`
  * 注：暂时还未考虑非正方形图像，故`INPUT_SHAPE`为正方形图形的边长，有待后续修改相关接口

* **构建模型** 这里是第一种构建方法。
  * 使用`Sequential`构建
  * 每一层是直线连接
  * **注意**：第一层无论是什么，层里的参数需要有`input_shape`，具体参数按照本例的写法即可
  * 层直接作为参数传入`Sequential`
  * 本例中的层顺序是：`Flatten`, `Dense(128, relu)`, `Dropout`, `Dense(softmax)`

* **检查保存** `self.check_save()`可以检测是否已经有之前训练过的模型保存点，并进行加载

* **测试部分（可选）** 在本例中测试部分中，用`cifar10`数据集图像尺寸构建了一个模型，并且调用了`summary()`函数来打印模型的一些层的情况。**注：为了不影响主程序调用，建议跟本例一样写在`if __name__ == "__main__":`里。另外，调试的时候如果出现`'NoneType' object has no attribute 'SAVE_EXIST'`，可以注释掉检查保存，或者引入ARGS。**

* **入库（重要）** 文件保存好之后，主函数并不能直接调用模型，需要入库。打开`__init__.py`文件，参照其他模型，添加一句代码即可。

  ```
  from .MLP import MLP
  ```
  注：`.`不能漏写，相对引包的标志
  
### 以MLP模型为例，第二种模型构建方法

```
self.model = Sequential()
self.model.add(Flatten(input_shape=(self.INPUT_SHAPE,     
                                    self.INPUT_SHAPE,
                                    self.INPUT_DEPTH)))
self.model.add(Dense(self.LOCAL_SIZE, activation='relu'))
self.model.add(Dropout(self.DROP_RATE))
self.model.add(Dense(self.NUM_CLASSES, activation='softmax'))
```

* **说明** 除了构建模型部分，其他都与第一种方法一致

* **构建模型** 这里是第二种构建方法。
  * 使用`Sequential`构建
  * 每一层是直线连接
  * **注意**：第一层无论是什么，层里的参数需要有`input_shape`，具体参数按照本例的写法即可
  * 使用`add()`方法添加层
  * 本例中的层顺序是：`Flatten`, `Dense(128, relu)`, `Dropout`, `Dense(softmax)`

### 以MLP模型为例，第三种模型构建方法

```
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
...
...
x_in = Input(shape=(self.INPUT_SHAPE, 
                    self.INPUT_SHAPE, 
                    self.INPUT_DEPTH))
x = Flatten()(x_in)
x = Dense(self.LOCAL_SIZE, activation='relu')(x)
x = Dropout(self.DROP_RATE)(x)
x = Dense(self.NUM_CLASSES, activation='softmax')(x)
self.model = Model(inputs=x_in, outputs=x, name='MLP')
```


* **说明** 除了引包和构建模型部分，其他都与第一种方法一致

* **构建模型** 这里是第三种构建方法。
  * 使用`Model`构建
  * 每一层可以不是直线连接，例：
    ```
    x = Conv2d(filters=self.INPUT_DEPTH, activation='relu')(x_in)
    x = Add()([x_in, x])
    ```
  * 使用`Input()`获取输入
  * 通过在层后面加入`(x)`来输入需要处理的值
  * 本例中的层是直线型的模型，顺序是：`Flatten`, `Dense(128, relu)`, `Dropout`, `Dense(softmax)`
  * 通过`Model`生成模型，需要填写输入输出，`name`可选。

## 创建数据集

>使用Tensorflow官方的Datasets创建

### 以mnist数据集为例

```
from .Packer import Packer
import tensorflow.keras.datasets as ds


class mnist(Packer):

  def __init__(self):
    super(mnist, self).__init__()
    self.NUM_TRAIN = 60000
    self.NUM_TEST = 10000
    self.NUM_CLASSES = 10
    self.IMAGE_SIZE = 28
    self.IMAGE_DEPTH = 1
    (self.train_images, self.train_labels), (self.test_images, self.test_labels) = ds.mnist.load_data()
    self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
    self.train_images = self.train_images.reshape((self.NUM_TRAIN, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_DEPTH))
    self.test_images = self.test_images.reshape((self.NUM_TEST, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_DEPTH))

```

* **引包** 引入`Packer`（必须），引入Tensorflow官方的datasets

* **构建类** 该类继承`Packer`类，需要注意的点有：
  * 类名和文件名最好一致，且为数据集本身的名字
  * 必须具有的参数：`NUM_CLASSES`, `IMAGE_SIZE`, `IMAGE_DEPTH`
  * 数据集的数据类型numpy数组，且应该为4D，[张数, 宽度, 高度, 深度]
    * 训练集图像：`train_images` 
    * 训练集标签：`train_labels` 
    * 测试集图像：`test_images` 
    * 测试集图像：`test_labels` 
  * 图像数字处理成 0~1之间的数
  * 注：暂时还未考虑非正方形图像，故`INPUT_SHAPE`为正方形的边长，有待后续修改相关接口

>自行创建数据集

* 待补充

