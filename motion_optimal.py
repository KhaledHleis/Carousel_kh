import numpy as np
import matplotlib.pyplot as plt

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

def draw_vector_field(xmin,xmax,ymin,ymax, R):
  X1, X2 = np.meshgrid(np.linspace(xmin,xmax,30), np.linspace(ymin,ymax,30))
  v1,v2 = phi(X1, X2,R)
  r = np.sqrt(v1**2 + v2**2)
  plt.quiver(X1, X2, v1/r, v2/r)

def motion_optimal(x, y, theta, K, u_bar, R):
  x_dot, y_dot = phi(x,y,R)
  theta_d = np.arctan2(y_dot,x_dot)
  d_theta = sawtooth(theta_d - theta)
  u1 = u_bar - K*d_theta
  u2 = u_bar + K*d_theta
  return u1, u2

def animate_euler(X0,tf,dt,K,u_bar,R):
  X = X0
  for t in np.arange(0,tf,dt):
    u1, u2 = motion_optimal(X[0], X[1], X[2], K, u_bar, R)
    v, w = (u1 + u2)/2, (u1 - u2)/2
    x_dot = v*np.cos(X[2])
    y_dot = v*np.sin(X[2])
    theta_dot = w
    X += np.array([x_dot, y_dot, theta_dot])*dt
    plt.scatter(X[0], X[1], color='red')
    plt.pause(0.01)

if __name__ == '__main__':

  draw_vector_field(-100,100,-100,100, 10)

  x0 = [30,30,2] #x,y,theta
  animate_euler(x0, 40, 0.5, 1, 5, 10)

  plt.show()