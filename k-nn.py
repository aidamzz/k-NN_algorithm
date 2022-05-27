import numpy as np 
from scipy import stats

#loading data

irises = np.load('irises.npy')
types = np.load('types.npy')
new_irises = np.load('new_irises.npy')

n, m = len(irises), len(new_irises)

#distance 

def calc_distance(new_points, points):
    m, n = len(new_points), len(points)
    d = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            d[i, j] = np.sum(np.square(new_points[i] - points[j]))
    return d

d = calc_distance(new_irises, irises)


# find k nearest 

k = 10

k_nearest = np.argpartition(d, k, axis=1)[:,:k]

#finding types of nearest

k_nearest_types = types[k_nearest]

# pridect types

predicted_types = stats.mode(k_nearest_types, axis=1).mode.reshape(m)

new_types = np.load('new_types.npy')
accuracy = np.sum(new_types == predicted_types) / m
print('Accuracy:', accuracy)
