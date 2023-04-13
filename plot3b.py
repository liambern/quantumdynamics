import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_file(V, gamma, beta=np.array([-0.2, -0.2, -0.2, -0.2, -0.2]), alpha=np.array([0.0, 0.0, 0.0])):
    file = open(str(V) + ',' + str(gamma) + str(alpha) + str(beta) + ".pkl", 'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file
V = 0.3
for gamma in [0.0, 0.001, 0.01]:
    current_i = load_file(V, gamma)[0]
    plt.plot(current_i, label='$\Gamma='+str(format(gamma, '.3f'))+'\mathtt{fs}^{-1}$')
plt.xlabel('Time [fs]')
plt.ylabel('Current [mA]')
plt.legend()
plt.savefig("3b_6.svg")
plt.clf()
