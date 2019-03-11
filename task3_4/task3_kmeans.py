# This script performs K-means clustering
#
# Ning Ma (n.ma@sheffield.ac.uk)
#
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from speechtech.frontend import compute_mfcc

# ==================================================
# CLUSTERING ROUTINES
# ==================================================
def dist(x, y):
	"""Calculate the Euclidean distance
	"""
	return np.sqrt(np.sum((x-y)**2,axis=1))

def kmeans_clustering(X, num_clusters, rseed=1):
	"""K-Means clustering

	Args:
		X: data to be clustered [num_samples x num_features]
		num_clusters: number of clusters
		rseed: random seed (default 1)

	Returns
		clusters: clustered labels for each sample in X
		centres: cluster centres
	"""
	num_samples = X.shape[0];

	# 1. Randomly choose num_clusters samples as cluster means
	np.random.seed(rseed)
	#====>>>>> centres = <FILL WITH YOUR CODE>

	clusters = np.zeros(num_samples)
	while True:
		# 2a. Assign data based on closest centre
		#====>>>>> <FILL WITH YOUR CODE>

		plt.scatter(X[:,1], X[:,2], s=30, c=clusters, cmap='viridis', alpha=0.8);
		plt.scatter(centres[:,1], centres[:,2], marker='*', s=100, c='red');
		plt.show()

		# 2b. Update cluster centres from means of each cluster
		#====>>>>> new_centres = <FILL WITH YOUR CODE>

		# Check for convergence
		if np.all(centres == new_centres):
			break
		centres = new_centres

	return clusters, centres


# ==================================================
# COMPUTE MFCCs for a list of vowels
# ==================================================
vowel_list = ['a', 'e', 'i', 'u'];
num_classes = len(vowel_list)
features = np.array([])
target_labels = []
for n in range(num_classes):
	print('Processing vowel {}'.format(vowel_list[n]))
	# Read waveform
	fs_hz, signal = wav.read('vowels/{0}.wav'.format(vowel_list[n]))
	mfcc = compute_mfcc(signal, fs_hz)
	num_frames = mfcc.shape[0]
	features = np.concatenate((features, mfcc)) if features.size else mfcc
	target_labels = np.append(target_labels, np.tile(n, num_frames))


# ==================================================
# K-MEANS CLUSTERING
# ==================================================

# --------------------------------------------------
# First we use K-means from scikit
# --------------------------------------------------
from sklearn.cluster import KMeans
kmeans = KMeans(num_classes, random_state=0, max_iter=20, algorithm='full')
clusters = kmeans.fit_predict(features)

# --------------------------------------------------
# Next try implementing your own K-means algorithm.
# Take note on the impact of initialisation. Use a
# different random seed and see the implications.
# --------------------------------------------------
# clusters, centres = kmeans_clustering(features, num_classes, rseed=1)


# ==================================================
# PLOTTING
# ==================================================
# Because k-means knows nothing about the identity of the cluster, the labels
# may be permuted. We can fix this by matching each learned cluster label with
# the true labels
from scipy.stats import mode
labels = np.zeros_like(clusters)
for n in range(num_classes):
	mask = (clusters == n)
	labels[mask] = mode(target_labels[mask])[0]

# Now we can check the confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(target_labels, labels)
plt.figure(1)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
	xticklabels=vowel_list, yticklabels=vowel_list, cmap="YlGnBu")
plt.xlabel('predicted vowel label')
plt.ylabel('target vowel label');

# Now we can scatter plot MFCCs. We use C2 and C3 instead of C1 and C2 as the
# scatter plot looks better
f1 = 1 # C2
f2 = 2 # C3
plt.figure(2)
plt.subplot(121)
plt.scatter(features[:,f1], features[:,f2], s=30, c=target_labels, cmap='viridis', alpha=0.8);
plt.title('true vowel labels')
plt.xlabel('C{}'.format(f1+1))
plt.ylabel('C{}'.format(f2+1))

plt.subplot(122)
plt.scatter(features[:,f1], features[:,f2], s=30, c=labels, cmap='viridis', alpha=0.8);
plt.title('predicted vowel labels')
plt.xlabel('C{}'.format(f1+1))
plt.ylabel('C{}'.format(f2+1))

plt.show()
