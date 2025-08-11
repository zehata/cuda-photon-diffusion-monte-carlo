# -*- coding: utf-8 -*-
# Adapted from original file located at
# https://colab.research.google.com/drive/18chJGmialCiQElLzFgLKMv-Dq_NNQev8

import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set the device globally
torch.set_default_device(device)

def photons(N):
  return torch.transpose(torch.stack((torch.zeros(N),torch.zeros(N),torch.ones(N),10*torch.ones(N))),0,1)

def evolve(vectors, packets):
  N = len(packets)

  transport_coefficient = 0.9999
  rng = 1-torch.rand(N)
  # print(rng)
  step_lengths = -torch.log(1-rng)/transport_coefficient
  # print(step_lengths)

  anisotropy = 0.25
  rng = 1-torch.rand(N)
  # print(rng)
  g = anisotropy
  mu = (1+g**2-((1-g**2)/(1-g+2*g*rng))**2)/(2*g)
  # print(mu)
  deflection_angles = torch.arccos(mu)
  # print(deflection_angles)

  rng = 1-torch.rand(N)
  psi = 2*np.pi*rng
  azimuthal_angles = psi
  # print(azimuthal_angles)

  rng = 0-torch.rand(N)

  sin_phis = torch.sin(azimuthal_angles)
  cos_phis = torch.cos(azimuthal_angles)
  sin_thetas = torch.sin(deflection_angles)
  cos_thetas = torch.cos(deflection_angles)

  vectors = torch.cat((vectors, torch.transpose(torch.stack((packets[:,0], packets[:,1], packets[:,2], step_lengths * sin_thetas * cos_phis, step_lengths * sin_thetas * sin_phis, -1 * step_lengths * cos_thetas)), 0, 1)))
  evolution = torch.transpose(torch.stack((step_lengths * sin_thetas * cos_phis, step_lengths * sin_thetas * sin_phis, -1 * step_lengths * cos_thetas, rng * packets[:,3])), 0, 1)
  packets = packets + evolution

  packets = packets[packets[:,2] < 0]

  return vectors, packets

packets = photons(1000)
vectors = torch.tensor([[0,0,0,0,0,0]])

for i in range(30):
  vectors, packets = evolve(vectors, packets)

soa = vectors.cpu().numpy()
X, Y, Z, U, V, W = zip(*soa)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=45, roll=0)
ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=.1)
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-20, 20])
plt.show()

def simulate():
  packets = photons(100000)
  vectors = torch.tensor([[0,0,0,0,0,0]])
  for i in range(1000):
    vectors, packets = evolve(vectors, packets)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
start = datetime.datetime.now()

simulate()

end = datetime.datetime.now()
print("Running on GPU took: " + str(end - start))

device = "cpu"
torch.set_default_device(device)
start = datetime.datetime.now()

simulate()

end = datetime.datetime.now()
print("Running on CPU took: " + str(end - start))