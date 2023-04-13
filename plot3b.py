import pickle
import matplotlib.pyplot as plt


for i in [0.000, 0.001, 0.010, 0.100, 1.000]:
    file = open(str(i) + ".pkl", 'rb')
    object_file = pickle.load(file)
    file.close()
    plt.plot(object_file[0], label='$\Gamma='+str(format(i, '.3f'))+'\mathtt{fs}^{-1}$')
plt.xlabel('Time [fs]')
plt.ylabel('Current [mA]')
plt.legend()
plt.savefig("3b_6.svg")
plt.clf()
