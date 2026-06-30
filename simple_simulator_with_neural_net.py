import numpy as np
import math
import matplotlib.pyplot as plt
from torch import Tensor

import neural_net
from datasets.get_dataset_v1 import get_input_line
from ddboatlib import pool_to_latlong, init_figure, draw_polygon, clear, draw_ddboat, ψ0
from neural_net import NeuralNetwork

# both NumPy (MKL) and PyTorch ship their own libiomp5md.dll, and Windows refuses to load it twice. It's not a bug in your code or the simulator.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def sawtooth(theta):
  return 2*np.arctan(np.tan(theta/2))

def phi_0(p1,p2):
  return -p1**3-p1*p2**2+p1-p2,-p2**3-p2*p1**2+p1+p2

def phi(x1,x2,R):
  D = np.array([[R,0],[0,R]])
  D_ = np.linalg.inv(D)
  z1, z2 = D_[0,0]*x1 + D_[0,1]*x2, D_[1,0]*x1 + D_[1,1]*x2
  w1,w2 = phi_0(z1,z2)
  v1,v2 = D[0,0]*w1 + D[0,1]*w2, D[1,0]*w1 + D[1,1]*w2
  r = np.sqrt(v1**2 + v2**2)
  return v1,v2

def motion_optimal(x, y, theta, K, u_bar, R):
  x_dot, y_dot = phi(x,y,R)
  theta_d = np.arctan2(y_dot,x_dot)
  d_theta = sawtooth(theta_d - theta)
  u1 = u_bar - K*d_theta
  u2 = u_bar + K*d_theta
  return u1, u2

def draw_vector_field(xmin,xmax,ymin,ymax, R):
  X1, X2 = np.meshgrid(np.linspace(xmin,xmax,30), np.linspace(ymin,ymax,30))
  v1,v2 = phi(X1, X2,R)
  r = np.sqrt(v1**2 + v2**2)
  plt.quiver(X1, X2, v1/r, v2/r)

def f(x, u):
   x,u=x.flatten(),u.flatten()
   θ,vx,vy,w,w1,w2=x[2],x[3],x[4],x[5],x[6],x[7]


   # Velocity x and y in the reference frame of the bassin
   dx=vx*np.cos(θ)-vy*np.sin(θ)
   dy=vx*np.sin(θ)+vy*np.cos(θ)
   dθ=w


   # Accelerations
   dvx=w*vy-p5*vx*abs(vx)+p3*(w1*abs(w1)+w2*abs(w2)) # longitudinal acceleration
   dvy=-w*vx-p6*vy*abs(vy) # lateral acceleration
   dw=-p7*w*abs(w) +p4*(w2*abs(w2)-w1*abs(w1)) # angular acceleration


   # Dynamics
   dw1=-p1*w1*abs(w1)+p2*u[0] # propeller velocity dynamics for the propeller 1
   dw2=-p1*w2*abs(w2)+p2*u[1] # propeller velocity dynamics for the propeller 2


   xdot=np.array([[dx],[dy],[dθ],[dvx],[dvy],[dw],[dw1],[dw2]])


   return xdot

if __name__ == '__main__':

    p1, p2, p3, p4, p5, p6, p7 = 0.07, 2200, 3.e-05, 15.e-05, 0.4, 5.0, 5.0

    ax = init_figure(-25, 25, -25, 25)
    x0, y0, θ0, vx0, vy0, w0, w10, w20 = 20, -20, 1, 10, 0, 0, 1, 1
    x = np.array([[x0, y0, θ0, vx0, vy0, w0, w10, w20]]).T
    dt = 0.01

    R = 10
    K, u_bar = 4, 20
    neural_net = NeuralNetwork("models/network_rl_torch.csv")
    for t in np.arange(0, 50, dt):
        clear(ax)
        draw_vector_field(-25,25,-25,25,R)
        # u1,u2 = motion_optimal(x[0,0], x[1,0], x[2,0], K, u_bar, R)
        inputs = Tensor(get_input_line(x[0,0], x[1,0], x[2,0]))
        u1, u2 =neural_net.forward(inputs)
        u1, u2 = u_bar * float(u1),u_bar * float(u2)
        # u2, u1 =neural_net.forward(inputs)
        u = np.array([[u1], [u2]])
        x = x + dt * f(x, u)  # Euler
        mx, my, θ, vx, vy, w, w1, w2 = list(x[0:8, 0])
        draw_ddboat(ax, mx, my, θ, w1, w2)
        plt.pause(0.0002)
    plt.pause(10)








