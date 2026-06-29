import numpy as np
import matplotlib.pyplot as plt

def phi_0(p1,p2):
  return -p1**3-p1*p2**2+p1-p2,-p2**3-p2*p1**2+p1+p2

def phi(x1,x2):
  global D
  D_ = np.linalg.inv(D)
  z1, z2 = D_[0,0]*x1 + D_[0,1]*x2, D_[1,0]*x1 + D_[1,1]*x2
  w1,w2 = phi_0(z1,z2)
  v1,v2 = D[0,0]*w1 + D[0,1]*w2, D[1,0]*w1 + D[1,1]*w2
  return v1,v2

def draw_vector_field(xmin,xmax,ymin,ymax):
  X1, X2 = np.meshgrid(np.linspace(xmin,xmax,30), np.linspace(ymin,ymax,30))
  v1,v2 = phi(X1, X2)
  R = np.sqrt(v1**2 + v2**2)
  plt.quiver(X1, X2, v1/R, v2/R)

r = 15
D = np.array([[r, 0], [0, r]])
draw_vector_field(-100,100,-100,100)
plt.show()