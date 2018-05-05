from matplotlib import pyplot as plt
import numpy as np

senses = np.loadtxt('sensesTestsense.txt')
a = np.array(senses)
fig = plt.plot(a[:,0], c = 'r')
fig = plt.plot(a[:,1], c = 'b')
# fig = plt.plot(a[:,2], c = 'y')
# fig = plt.plot(a[:,3], c = 'o')
# fig = plt.plot(a[:,4], c = 'r')
# fig = plt.plot(a[:,5], c = 'b')
# fig = plt.plot(a[:,6], c = 'g')
# fig = plt.plot(a[:,7], c = 'h')
plt.show()