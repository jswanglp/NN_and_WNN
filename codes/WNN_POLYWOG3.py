# 小波神经网络
# 程序实现包含一个小波隐藏层的小波神经网络,小波函数为 POLYWOG3 函数
# 该小波神经网络无论是收敛速度，还是拟合能力能力都比含双隐层的普通神经网络要强
# 详见 NN.py，NN.ipynb，Wavelet_NN.py 和 Wavelet_NN.ipynb
# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np 
import time

# 计时装饰器
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print('Training time is :{:.2f} s.'.format(end_time - start_time))
    return wrapper

# 定义网络结构类
class WaveletNeuralNet(object):
    # 初始化神经网络，sizes是神经网络的层数和每层神经元个数  
    def __init__(self, sizes, k, name='morlet wavelet'):
        self.sizes_ = sizes
        self.num_layers_ = len(sizes)  # 层数
        if self.num_layers_ > 3:
            print('ERROR!')
        self.num_nuerals_ = sizes[1]
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # w_、b_初始化为正态分布随机数
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]
        self.t_ = np.random.randint(2, 15, (self.num_nuerals_, 1))
#         self.t_ = np.random.normal(5, 2., (self.num_nuerals_, 1))
        self.s_ = 2 * np.random.randn(self.num_nuerals_, 1)
        self.k = k
        if name == 'morlet wavelet' or name == 'POLYWOG3 wavelet':
            self.name = name
        else: self.err_print()
    
    # 错误打印
    def err_print(self):
        print("Unsupported wavelet!\nOnly support 'morlet wavelet' and 'POLYWOG3 wavelet'.")
    
    # 标签转化
    def one_hot(self, x, num_classes):
        x = x.flatten().astype('uint8')
        m = x.shape[0]
        x_onehot = np.zeros((m, num_classes))
        for i in range(m):
            x_onehot[i, x[i]] = 1
        return x_onehot
        
    # Sigmoid函数，S型曲线，
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # Sigmoid函数的导函数
    def sigmoid_der(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    # morlet小波母函数
    def phi(self, z, t=1, s=0):
        k = self.k
        z_ = (z - s) / t
        return np.cos(k * z_) * np.exp(-z_**2 / 2.)
    
    # 小波函数导数
    def phi_der(self, z, t=1, s=0):
        k = self.k
        z_ = (z - s) / t
        return (-k * np.sin(k * z_) * np.exp(-z_**2 / 2) - z_ * np.cos(k * z_) * np.exp(-z_**2 / 2)) / t
    
    # POLYWOG3 小波母函数
    def polywog3(self, z, t=1, s=0):
        k = self.k
        z_ = (z - s) / t
        c_exp = np.exp(-.5 * np.power(z_, 2))
        return k * (np.power(z_, 4) - 6 * np.power(z_, 2) + 3) * c_exp
    
    # POLYWOG3 小波导函数
    def polywog3_der(self, z, t=1, s=0):
        k = self.k
        z_ = (z - s) / t
        c_exp = np.exp(-.5 * np.power(z_, 2))
        return k * (10 * np.power(z_, 3) - np.power(z_, 5) - 15 * z_) * c_exp

    def feedforward(self, x): # 前向
        if self.name == 'morlet wavelet':
            func = self.phi
        elif self.name == 'POLYWOG3 wavelet':
            func = self.polywog3
        else: self.err_print()
        n = self.w_[0].shape[1]
        x = x.reshape(n, -1)
        x1 = func(np.dot(self.w_[0], x) + self.b_[0], self.t_, self.s_)
        x2 = self.sigmoid(np.dot(self.w_[1], x1) + self.b_[1])
        return x2
    
    # 反向传播
    def backprop(self, x, y):
        if self.name == 'morlet wavelet':
            func = self.phi
            func_der = self.phi_der
        elif self.name == 'POLYWOG3 wavelet':
            func = self.polywog3
            func_der = self.polywog3_der
        else: self.err_print()
        b_new = [np.zeros(b.shape) for b in self.b_]
        w_new = [np.zeros(w.shape) for w in self.w_]
        t_new = self.t_
        s_new = self.s_
        activation = x
        activations = [x]  # activations代表着每层的输出
        zs = []  # zs代表着每层的输入，即前层输出与权重的和
        z = np.dot(self.w_[0], activation) + self.b_[0]
        zs.append(z)
        activation = func(z, t_new, s_new)
        activations.append(activation)
        z = np.dot(self.w_[1], activation) + self.b_[1]
        zs.append(z) 
        activation = self.sigmoid(z)
        activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_der(zs[-1])
        b_new[-1] = delta
        w_new[-1] = np.dot(delta, activations[-2].transpose())
        
        delta_last = delta.copy()
        z = zs[-2]
        sp = func_der(z, t_new, s_new)
        delta = np.dot(self.w_[-1].transpose(), delta_last) * sp
        b_new[-2] = delta
        w_new[-2] = np.dot(delta, activations[-3].transpose())
        sp_t = -.5 * t_new**-1.5 * func((z-s_new) / t_new) - t_new**-2.5 * (z - s_new) * func_der((z - s_new) / t_new)
        sp_s = -t_new**-1.5 * func_der((z-s_new) / t_new)
        # t_new = np.dot(self.w_[-1].transpose(), delta_last)*sp_t # loss函数对小波函数缩放系数的偏导
        # s_new = np.dot(self.w_[-1].transpose(), delta_last)*sp_s # loss函数对小波函数平移系数的偏导
        
        t_new = delta * sp_t # loss函数对小波函数缩放系数的偏导
        s_new = delta * sp_s # loss函数对小波函数平移系数的偏导
        
        return (b_new, w_new, t_new, s_new)

    # 更新权值w，偏移b，缩放因子t，偏移因子s
    def update_mini_batch(self, mini_batch, lr):
        b_new = [np.zeros(b.shape) for b in self.b_]
        w_new = [np.zeros(w.shape) for w in self.w_]
        a, b = mini_batch[:, :-1], self.one_hot(mini_batch[:, -1], num_classes=10)
        n = np.float(mini_batch.shape[0])
        for i in range(int(n)):
            x, y = a[i, :].reshape(-1, 1), b[i, :].reshape(-1, 1)
            delta_b_new, delta_w_new, t_new, s_new = self.backprop(x, y)
            b_new = [nb + dnb for nb, dnb in zip(b_new, delta_b_new)]
            w_new = [nw + dnw for nw, dnw in zip(w_new, delta_w_new)]
        self.w_ = [w - lr * nw for w, nw in zip(self.w_, w_new)]
        self.b_ = [b - lr * nb for b, nb in zip(self.b_, b_new)]
        self.t_ = self.t_ - lr * t_new
        self.s_ = self.s_ - lr * s_new
    
    # training_data是训练数据(x, y), epochs是训练次数, mini_batch_size是每次训练样本数, lr是learning rate，step是展示的迭代间隔
    @timer
    def SGD(self, training_data, epochs=50, mini_batch_size=32, lr=.1, step=10):
        assert type(step) == int, 'Step must be a integer.'
    
        n = training_data[0].shape[0]
        for j in range(epochs):
            ss = np.hstack((training_data[0],training_data[1].reshape(n, -1)))
            np.random.shuffle(ss)
            mini_batches = [ss[k:k + mini_batch_size, :] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            accur = self.evaluate(training_data) * 100
            mse_loss = self.mse_loss(training_data)
            if (j + 1) % step == 0 or j == 0:
                print("Epoch {0}, mse_loss: {1:.4f}, accury on the training set :{2:.2f}{3}".format(j+1, mse_loss, accur, '%'))
            # print("Epoch {0}: {1} / {2}".format(j, self.evaluate(training_data), n))
    
    # 计算正确率
    def evaluate(self, data):
        x_t, x_label = data
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in zip(list(x_t), list(x_label))]
        acc = sum(int(x == y) for (x, y) in test_results) / x_t.shape[0]
        return acc
    
    # mse_loss的导数
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    # mse_loss
    def mse_loss(self, training_data):
        x_t,x_label = training_data
        test_results = [.5 * norm(self.feedforward(x).flatten() - self.one_hot(y, num_classes=10))**2
                        for (x, y) in zip(list(x_t), list(x_label))]
        return np.array(test_results).mean()
    
    # 预测
    def predict(self, data):
        data = data.reshape(-1, self.sizes_[0])
        value = np.array([np.argmax(net.feedforward(x)) for x in data], dtype='uint8')
        return value
    
    # 保存训练模型
    def save(self):
        pass  # 把_w和_b保存到文件(pickle)  
    
    def load(self):
        pass
        
if __name__ == '__main__':

    mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)
    
    num_classes = 10
    training_data = mnist.train.next_batch(5000)
    testing_data = mnist.test.next_batch(1000)
    net = WaveletNeuralNet([784, 128, num_classes], k=1.75, name='symlets wavelet')
    # >> Unsupported wavelet!
    # >> Only support 'morlet wavelet' and 'POLYWOG3 wavelet'.

    # ----------------------Morlet wavelet-----------------------------------------
    net = WaveletNeuralNet([784, 128, num_classes], k=1.75, name='morlet wavelet')
    title_name = net.name.capitalize() + r', $cos(%.2fx)e^{\frac{-x^{2}}{2}}$' % net.k
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111)
    x = np.linspace(-4, 4, 201, endpoint=True)
    y = net.phi(x, t=1, s=0)
    ax.plot(x, y, label='morlet wavelet')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(title_name, fontsize=14)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel(r'$\psi (x)$', rotation='horizontal', fontsize=14)
    ax.yaxis.set_label_coords(-.05,1.02)

    # # Derivative function
    # y_d = net.phi_der(x, t=1, s=0)
    # ax_d = fig.add_subplot(122)
    # ax_d.plot(x, y_d, label='derivative')
    # ax_d.legend(loc='upper right', fontsize=12)
    # ax_d.set_title('Derivative of morlet wavelet', fontsize=14)
    # ax_d.set_xlabel('x', fontsize=14)
    # ax_d.set_ylabel("${\psi}'(x)$", rotation='horizontal', fontsize=14)
    # ax_d.yaxis.set_label_coords(-.05,1.02)

    net.SGD(training_data, epochs=200, mini_batch_size=32, lr=.1, step=20)

    # >> Epoch 1, mse_loss: 0.3483, accury on the training set :48.18%
    # >> Epoch 20, mse_loss: 0.0474, accury on the training set :93.06%
    # >> Epoch 40, mse_loss: 0.0246, accury on the training set :95.72%
    # >> Epoch 60, mse_loss: 0.0198, accury on the training set :96.48%
    # >> Epoch 80, mse_loss: 0.0170, accury on the training set :96.86%
    # >> Epoch 100, mse_loss: 0.0160, accury on the training set :96.96%
    # >> Epoch 120, mse_loss: 0.0153, accury on the training set :97.12%
    # >> Epoch 140, mse_loss: 0.0143, accury on the training set :97.28%
    # >> Epoch 160, mse_loss: 0.0139, accury on the training set :97.38%
    # >> Epoch 180, mse_loss: 0.0135, accury on the training set :97.44%
    # >> Epoch 200, mse_loss: 0.0133, accury on the training set :97.54%
    # >> Training time is :1207.57 s.

    net.evaluate(testing_data)
    # >> 0.902

    # ----------------------POLYWOG3 wavelet-----------------------------------------
    net = WaveletNeuralNet([784, 128, num_classes], k=1. / 3, name='POLYWOG3 wavelet')
    title_name = net.name + r', $%.2f(x^{4}-6x^{2}+3)e^{\frac{-x^{2}}{2}}$' % net.k
    fig = plt.figure(2, figsize=(8, 6))
    ax = fig.add_subplot(111)
    x = np.linspace(-4, 4, 201, endpoint=True)
    y = net.polywog3(x, t=1, s=0)
    ax.plot(x, y, label='POLYWOG3 wavelet')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(title_name, fontsize=14)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel(r'$\psi (x)$', rotation='horizontal', fontsize=14)
    ax.yaxis.set_label_coords(-.05,1.02)

    # # Derivative function
    # y_d = net.polywog3_der(x, t=1, s=0)
    # ax_d = fig.add_subplot(122)
    # ax_d.plot(x, y_d, label='derivative')
    # ax_d.legend(loc='upper right', fontsize=12)
    # ax_d.set_title('Derivative of POLYWOG3 wavelet', fontsize=14)
    # ax_d.set_xlabel('x', fontsize=14)
    # ax_d.set_ylabel("${\psi}'(x)$", rotation='horizontal', fontsize=14)
    # ax_d.yaxis.set_label_coords(-.05,1.02)

    net.SGD(training_data, epochs=200, mini_batch_size=32, lr=.1, step=20)

    # >> Epoch 1, mse_loss: 0.2867, accury on the training set :63.04%
    # >> Epoch 20, mse_loss: 0.0103, accury on the training set :98.26%
    # >> Epoch 40, mse_loss: 0.0071, accury on the training set :98.76%
    # >> Epoch 60, mse_loss: 0.0060, accury on the training set :98.90%
    # >> Epoch 80, mse_loss: 0.0053, accury on the training set :99.00%
    # >> Epoch 100, mse_loss: 0.0048, accury on the training set :99.08%
    # >> Epoch 120, mse_loss: 0.0047, accury on the training set :99.10%
    # >> Epoch 140, mse_loss: 0.0047, accury on the training set :99.10%
    # >> Epoch 160, mse_loss: 0.0043, accury on the training set :99.22%
    # >> Epoch 180, mse_loss: 0.0039, accury on the training set :99.26%
    # >> Epoch 200, mse_loss: 0.0038, accury on the training set :99.26%
    # >> Training time is :1379.53 s.

    net.evaluate(testing_data)
    # >> 0.936