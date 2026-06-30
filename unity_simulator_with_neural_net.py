import json
import socket
import numpy as np

import neural_net
from datasets.get_dataset_v1 import get_input_line
from neural_net import NeuralNetwork
# from torch import Tensor



UNITY_IP = "10.125.4.72"
COMMAND_PORT = 5005
POSE_PORT = 5006

command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pose_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pose_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
pose_socket.bind(("", POSE_PORT))


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

def send_command(u1, u2):
    command = json.dumps({"u1": u1, "u2": u2}).encode("utf-8")
    command_socket.sendto(command, (UNITY_IP, COMMAND_PORT))


def receive_pose():
    data, _ = pose_socket.recvfrom(4096)
    pose = json.loads(data.decode("utf-8"))
    x = pose.get("x")
    y = pose.get("y")
    cos_value = pose.get("cos")
    sin_value = pose.get("sin")
    return x, y, cos_value, sin_value


if __name__ == "__main__":
    neural_net = NeuralNetwork("models/network_rl_torch.csv")
    while True:  # Control loop

        x, y, cos_value, sin_value = receive_pose()
        theta = np.arctan2(sin_value, cos_value)
        inputs = get_input_line(x, y, theta)
        u1, u2 =neural_net.forward(inputs)
        # print(u1,u2)
        # u1, u2 = motion_optimal(x, y, theta, K=0.1, u_bar=0.5, R=10)
        send_command(u1,u2)
        # send_command(-1,-1)

         