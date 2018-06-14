# 简介
DCGAN on MNIST

学习GAN的练习代码,一些心得参见[博文](https://blog.csdn.net/z0n1l2/article/details/80693665) 欢迎讨论指教

# 参考
源码来自 https://github.com/dbsheta/dcgan_face_generation
此处只是略作修改,成为针对MNIST的DCGAN


# 实验结果
训练集采用mnist的train部分,5w张图片, batch_size=100

初始G网路的输出   
![初始](https://github.com/z01nl1o02/toy-gan/blob/master/result/Gnet-output-%E5%88%9D%E5%A7%8B.jpg)     

训练中间G网络的输出,已经可以看出一些眉目了    
![中间](https://github.com/z01nl1o02/toy-gan/blob/master/result/Gnet-output-%E4%B8%AD%E9%97%B4%E8%BF%87%E7%A8%8B.jpg)     

大概7w个iteration后的结果   
![结果](https://github.com/z01nl1o02/toy-gan/blob/master/result/Gnet-output-%E6%9C%80%E7%BB%88%E7%BB%93%E6%9E%9C.jpg)  



