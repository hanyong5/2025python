import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



fruits= np.load('data/fruits_300.npy')
# fruits_2d = fruits.reshape(300,100*100)
fruits_2d = fruits.reshape(-1,100*100)
# print(fruits_2d.shape)

km = KMeans(n_clusters=3,random_state=42)
km.fit(fruits_2d)

print(km.labels_)
print(np.unique(km.labels_,return_counts=True))


def draw_fruits(arr, ratio=1):
    n = len(arr) 

    rows = int(np.ceil(n/10))

    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,figsize=(colsratio, rowsratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i10 + j < n:
                axs[i, j].imshow(arr[i10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()