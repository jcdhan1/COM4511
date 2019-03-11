# This script performs frame-based vowel classification using 
# Gaussian Mixture Models

# =================================================================
# GENERAL IMPORTS
# =================================================================
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from speechtech.frontend import compute_mfcc
from sklearn.mixture import GaussianMixture


# =================================================================
# GLOBAL VARIABLES
# =================================================================
VOWELS = ['a', 'e', 'i', 'u'];
NUM_VOWELS = len(VOWELS)
DATA_DIR = 'vowels'
NUM_MIXTURES = 1  # number of Gaussian mixtures per GMM
CMN = False


# =================================================================
# TRAINING ROUTINE 
# =================================================================
def train_GMMs(features, target_labels):

	print('number of training frames = {}'.format(len(target_labels)))
	
	gmm_set = [GaussianMixture(n_components=NUM_MIXTURES, covariance_type='diag', 
		init_params='kmeans', max_iter=100) for gmm_id in range(NUM_VOWELS)]

	for vowel_id in range(NUM_VOWELS):
		
		# Extract features for a vowel
		vowel_features = features[target_labels==vowel_id]

		# Train GMM
		print('Traning vowel model %s' % VOWELS[vowel_id], vowel_features.shape)
		# ====>>>> FILL WITH YOUR CODE
		# ====>>>> TRAINED GMMS SHOULD BE SAVED IN "gmm_set"

	return gmm_set


# =================================================================
# EVALUATION ROUTINE 
# =================================================================
def eval_GMMs(gmm_set, features, target_labels):

	print('number of test frames = {}'.format(len(target_labels)))

	# Performa classification for each frame
	# ====>>>> FILL WITH YOUR CODE
	# ====>>>> SAVE RECOGNITION OUTPUT IN "rec_labels"
	# ====>>>> USE GMM INDEX IN "gmm_set" AS ITS LABEL [0,1,2,3]

	# Compute accuracy
	acc = np.mean(target_labels == rec_labels)

	print('Vowel frame accuracy is {0}%%'.format(acc*100))

	# Now we can check the confusion matrix
	from sklearn.metrics import confusion_matrix
	mat = confusion_matrix(target_labels, rec_labels)
	plt.figure(1)
	sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, 
		xticklabels=VOWELS, yticklabels=VOWELS, cmap="YlGnBu")
	plt.xlabel('predicted vowel label')
	plt.ylabel('target vowel label');

	# Now we can scatter plot MFCCs. We use C2 and C3 instead of C1 and C2 as the 
	# scatter plot looks better
	f1 = 1 # C2
	f2 = 2 # C3
	plt.figure(2)
	plt.subplot(121)
	plt.scatter(features[:,f1], features[:,f2], s=30, c=target_labels, cmap='viridis', alpha=0.8);
	plt.title('target vowel labels')
	plt.xlabel('C{}'.format(f1+1))
	plt.ylabel('C{}'.format(f2+1))

	plt.subplot(122)
	plt.scatter(features[:,f1], features[:,f2], s=30, c=rec_labels, cmap='viridis', alpha=0.8);
	plt.title('predicted vowel labels')
	plt.xlabel('C{}'.format(f1+1))
	plt.ylabel('C{}'.format(f2+1))

	plt.show()

	return acc


# =================================================================
# MAIN FUNCTION 
# =================================================================
def main():

	# Load all vowel signals and compute MFCCs
	# Accumulate MFCCs for all frames and corresponding labels
	features = np.array([])
	target_labels = []
	for vowel_id in range(NUM_VOWELS):
		print('Processing vowel {}'.format(VOWELS[vowel_id]))
		# Read waveform
		fs_hz, signal = wav.read('vowels/{0}.wav'.format(VOWELS[vowel_id]))
		mfcc = compute_mfcc(signal, fs_hz)
		num_frames = mfcc.shape[0]
		features = np.concatenate((features, mfcc)) if features.size else mfcc
		target_labels = np.append(target_labels, np.tile(vowel_id, num_frames))

	# Train GMMs
	gmm_set = train_GMMs(features, target_labels)
	
	# NOTE: WE ARE TESTING THE GMMS USING THE TRAINING SET
	eval_GMMs(gmm_set, features, target_labels)


# =================================================================
if __name__ == '__main__':
	main()

