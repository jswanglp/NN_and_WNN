# NN_and_WNN

> 业精于勤，荒于嬉；行成于思，毁于随<a href='#fn1' name='fn1b'><sup>[1]</sup></a>。  

一个通过[反向传播算法](https://en.wikipedia.org/wiki/Backpropagation)来实现[神经网络](https://en.wikipedia.org/wiki/Artificial_neural_network)与[小波神经网络](https://pdfs.semanticscholar.org/0c8b/e141c9092ed389b9931ac09ec2e852d437c6.pdf)的 `repo`，由于未使用到 `GPU` 加速， 当网络层数较多时会导致训练比较慢，训练集也只是截取了 [mnist](http://yann.lecun.com/exdb/mnist/) 手写数据集中的 `5000` 张图片，测试集则选择了 `1000` 张。  

需要安装的库包括：
```
TensorFlow 1.12.0 (如果已下载 `mnist` 手写数据集则不需要)  
numpy 1.15.4  
matplotlib 2.0.2  
```

**神经网络 (Neural Network)** 程序实现包含 `2` 个隐藏层的神经网络，激活函数为 `sigmoid` 函数，运行结果的笔记保存至 `jupyter notebook` 文件。  
- [code](./codes/NN.py)  
- [jupyter notebook](./notebooks/NN.ipynb)  

-----
**脚注 (Footnote)**

<a name='fn1'>[1]</a>：[进学解 -- 韩愈](https://so.gushiwen.org/shiwenv_94a69d56db65.aspx)  

<a href='#fn1b'><small>↑Back to Content↑</small></a>