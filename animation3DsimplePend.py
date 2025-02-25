import vpython
from vpython import *
import numpy as np

x1, y1, z1 = np.load('.\\data\\3Dpen.npy')
ball1 = vpython.sphere(color = color.green, radius = 0.3, make_trail=True, retain=20)

rod1 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.05)
base  = box(pos=vector(0,-11,0),axis=vector(1,0,0),
            size=vector(10,0.5,10) )
s1 = cylinder(pos=vector(0,-11.5,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))


print('Start')
i = 0
while True:
    rate(300)
    i = i + 1
    i = i % len(x1)
    ball1.pos = vector(x1[i], z1[i], y1[i])
    rod1.axis = vector(x1[i], z1[i], y1[i])
    # s1.pos = vector(x1[i], -3.99, y1[i])