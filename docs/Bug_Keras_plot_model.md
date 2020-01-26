### 环境

* 系统版本：Win10 专业版1809
* 设备：Laptop
* Tensorflow版本：Tensorflow-gpu 1.14
* Python版本：Python3.6.8-Anaconda custom(64-bit)
* CUDA版本：CUDA-V10.0.130
* cuDNN版本：cuDNN-7.5.0
* GPU：Nvidia GTX1050Ti

### BUG描述

在tensorflow里使用keras的utils中的plot_model的时候，出现错误

### 日志

```
from keras.utils.visualize_util import plot
  File "D:\Anaconda3\lib\site-packages\keras\utils\visualize_util.py", line 13, in <module>
    raise RuntimeError('Failed to import pydot. You must install pydot'
RuntimeError: Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.
```

### 相关代码

```python
from tensorflow.python.keras.utils import plot_model
plot_model(model, to_file='LeNet.png')
```

### 问题解决

#### 2020.01.26 更新

安装顺序：

```
安装graphviz.msi
conda install graphviz
conda install pydot
```

不需要再改代码。如果还是报错，可能还需要试试

```
添加PATH: graphviz/bin
```

---

参考链接：[issue 3210](https://github.com/keras-team/keras/issues/3210)

```
安装graphviz的msi文件
添加PATH: graphviz/bin
pip install pydot_ng
conda install graphviz
修改代码
```

```
# tensorflow.python.keras.utils.vis_utils.py
L36 
	import pydot
	->import pydot_ng as pydot
# pydot_ng.__init__.py
L482-L560
	注释掉
L572-L579
	注释掉
L581
	->path = r"C:\Program Files (x86)\Graphviz2.38\bin"
```

### bug原因

最新的pydot中没有`find_graphviz()`函数，而`pydot_ng`还有，因此安装该版本。另外需要修改部分代码，让`pydot`找到`graphviz`目录。

