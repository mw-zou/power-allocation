import numpy as np
import matplotlib.pyplot as plt
import csv
from pylab import *

plt.figure()
plt.title("value of the equilibrium game ")
# x=[0, 1, 0.5,0]
# y=[0, 0, a,0]
x = [1/4, 1/8, 1/16, 1/32, 1/64]
y = [0.778, 0.461, 0.253, 0.133,0.068]

plt.xlabel("Segmentation granularity c")
#  plt.xlabel("iteration")
plt.ylabel("value of the equilibrium game")
plt.plot(x, y, "b*", linewidth=2, markersize=10)
#plt.plot(x, y)  ###画出x和y之间的连线,对应关系
plt.gca().invert_xaxis()###将x轴反向显示
plt.show()

