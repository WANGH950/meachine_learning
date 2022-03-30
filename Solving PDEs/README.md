# 偏微分方程的神经网络解法（毕业论文源码）
王恒，兰州大学，数学与统计学院，2018数学基地班

## 环境配置
python==3.7.5 torch==1.8.1

## deepBSDE
复现了文献[1]中的deepBSDE方法, 并从万有近似定理的角度提出了deepBSDE的优化方法deepBSDE++. 以Black-Scholes方程为例, 实现了两种方法.
详见"./black-scholes/", "blackScholes.py"中定义了Black-Scholes方程类以及训练器, "deepBSDE.py"和"deepBSDEplus.py"分别实现了deepBSDE和deepBSDE++算法.

## PINNs
复现了文献[2]中的连续时间物理信息神经网络(PINN)模型, 在聚合物扩散的Feynman-Kac方程上进行实验, 分别求解了向前和向后方程.
详见"./polymer-diffusion/", "polymerDiffusion.py"中定义了聚合物扩散的Feynman-Kac方程类以及模型训练器, "forward.py"和"backward.py"分别实现了向前和向后方程的PINN.

## 参考文献
[1] Han J, Arnulf J, Weinan E. Solving high-dimensional partial differential equations using deep learning[J]. Proceedings of the National Academy of Sciences of the United States of America, 2018, 115(34): 8505-8510.

[2] M. Raissi, P. Perdikaris, G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707.