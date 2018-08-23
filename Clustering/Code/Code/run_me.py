# Import modules

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import misc

def read_scene():
	data_x = misc.imread('../../Data/umass_campus_100x100x3.jpg')

	return (data_x)

if __name__ == '__main__':
	
	################################################
	# K-Means

	data_x = read_scene()
	print('X = ', data_x.shape)

	data_xx = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
	print('Flattened image = ', data_xx.shape)

	print('Implement k-means here ...')

	fig1, axarr = plt.subplots(3, 3, figsize=(12, 12))  # rows, columns, figure size

	# plot the original image
	axarr[0, 0].set_title("Original Image")
	axarr[0, 0].set_xticks([])  # tell the ticks to go away
	axarr[0, 0].set_yticks([])
	axarr[0, 0].imshow(data_x)
	av = np.mean(data_xx, axis=0)

	num_clusters = np.array([2, 5, 10, 25, 50, 75, 100, 200])
	err_km = np.zeros((num_clusters.size))
	per_var_ex = np.zeros((num_clusters.size, 3));
	for i in range(len(num_clusters)):

		# Do your clustering
		out = np.uint8(np.zeros((10000, 3)))
		n = num_clusters[i]
		kmeans = KMeans(n_clusters=n, random_state=0).fit(data_xx)
		pred = kmeans.labels_
		cents = kmeans.cluster_centers_

		# Reconstruct the image using your clusters
		for x in range(0, n):
			out[np.argwhere(pred == x)] = np.uint8(cents[x])

		recon_error = np.sqrt(np.mean((out - data_xx) ** 2))
		print('K-means, number of clusters:', n)
		print('K-means, reconstruction error:', recon_error)
		err_km[i] = recon_error

		recon_img = out.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
		# I called mine recon_image. It should be the same .shape as flattened_image

		row = np.int(np.floor((i + 1) / 3))
		col = np.int(i - 3 * row + 1)
		# Plot the reconstructed image
		# I made a helper function to get the row,col from the iteration variable i
		axarr[row, col].set_xticks([])  # tell the ticks to go away
		axarr[row, col].set_yticks([])
		axarr[row, col].set_title("K-means: " + str(n))
		axarr[row, col].imshow(misc.toimage(recon_img.reshape(data_x.shape)))
		exp_var = np.sum((out - av) ** 2, axis=0)
		unexp_var = np.sum((out - data_xx) ** 2, axis=0)
		per_var_ex[i, :] = exp_var / unexp_var
plt.show()

### Elbow plot
plt.figure()
plt.plot(num_clusters, np.mean(per_var_ex,axis=1))
#ax1.ylabel('Percentage of Variance Explained ')
#ax1.xlabel('Number of Clusters')
plt.show()

###### Do HAC
from sklearn.cluster import AgglomerativeClustering
plt.figure()
fig2, axarr = plt.subplots(3, 3, figsize=(12, 12))  # rows, columns, figure size

# plot the original image
axarr[0, 0].set_title("Original Image")
axarr[0, 0].set_xticks([])  # tell the ticks to go away
axarr[0, 0].set_yticks([])
axarr[0, 0].imshow(data_x)

num_clusters = np.array([2, 5, 10, 25, 50, 75, 100, 200])
# num_clusters=np.array([25,200])
err_hac = np.zeros((num_clusters.size))
per_var_ex = np.zeros((num_clusters.size, 3));
av = np.mean(data_xx, axis=0)

for i in range(len(num_clusters)):

    # Do your clustering
    out2 = np.uint8(np.zeros((10000, 3)))
    out = np.uint8(np.zeros((10000, 3)))
    n = num_clusters[i]
    print('HAC, number of clusters:', n)
    myHAC = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    out2 = myHAC.fit(data_xx)
    pred = out2.labels_

    ### cents contains the centroids
    cents = np.zeros((n, 3))
    exp_var = np.zeros((1, 3))

    ### find the centroids and reconstruct the image using your clusters
    for x in range(0, n):
        cents[x, :] = np.mean(data_xx[np.argwhere(pred == x)], axis=0)
        out[np.argwhere(pred == x)] = np.uint8(cents[x])
        nums = len(np.argwhere(pred == x))

    recon_error = np.sqrt(np.mean((out - data_xx) ** 2))
    print('HAC, reconstruction error:', recon_error)
    err_hac[i] = recon_error

    recon_img = out.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
    # I called mine recon_image. It should be the same .shape as flattened_image

    row = np.int(np.floor((i + 1) / 3))
    col = np.int(i - 3 * row + 1)
    # Plot the reconstructed image
    # I made a helper function to get the row,col from the iteration variable i
    axarr[row, col].set_xticks([])  # tell the ticks to go away
    axarr[row, col].set_yticks([])
    axarr[row, col].set_title("HAC: " + str(n))
    axarr[row, col].imshow(misc.toimage(recon_img.reshape(data_x.shape)))
    exp_var = np.sum((out - av) ** 2, axis=0)
    unexp_var = np.sum((out - data_xx) ** 2, axis=0)
    per_var_ex[i, :] = exp_var / unexp_var
    plt.show