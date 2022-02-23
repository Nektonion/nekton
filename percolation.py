#cd /home/nekton/miniconda3/Python_course/Perco/perco.py
import matplotlib.pyplot as plt
import numpy as np
import random
from random import randrange
from scipy.ndimage import measurements
p_max = 100
p = np.linspace(0, p_max-1, num=100)
#   for percent in p:
    #print(percent)
runs = 10000
# runs = amount of matrices
# n is matrix dimension
N = np.array([10, 15, 20])

for n in N:
    w = np.zeros(p_max)
    w_norm = np.zeros(p_max)
    for percent in p:
        for run in range(0, runs):
            matrix = np.zeros((n,n))
            # filling 0-matrix with 1
            number_of_ones = round(percent * n * n / p_max)
            l = list(range(1, n**2 + 1))
            random.shuffle(l)
            for lucky in range(0, number_of_ones):
                i = (l[lucky] - 1) // n
                j = l[lucky] % n
                matrix[i][j] = 1
            #print(run, percent)
            #print(matrix)
            labeled_array, clusters = measurements.label(matrix)
            cluster_check = np.intersect1d(labeled_array[0,:],labeled_array[-1,:])
            perc = cluster_check[np.where(cluster_check > 0)]
            if (len(perc) > 0):
                w[round(percent)] += 1
        w_norm = w / runs / p_max
        #print(w, w_norm)
    plt.plot(p, w_norm, label='w(p)')
    plt.xlabel('p')
    plt.ylabel('w')
plt.show()
