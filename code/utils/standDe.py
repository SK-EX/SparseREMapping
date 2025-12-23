import matplotlib.pyplot as plt
import numpy as np
import math
x = np.linspace(-4,4,50)#设置画图范围
#sigma = 0.5*math.sqrt(2*math.pi)
sigma = 4  #方差设置为2
u = 0 #均值设置为1
y =( 1/(math.sqrt(2*math.pi)*sigma))*np.exp(-(x-u)**2/2*sigma**2)
# y = x/(1+np.exp( -x ))  #sigmoid
plt.figure()
plt.plot(x,y)
plt.show()