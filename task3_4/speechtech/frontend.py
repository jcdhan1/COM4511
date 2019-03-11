
import numpy as np
from scipy.fftpack import dct

def freq2mel(freq):
	"""Convert Frequency in Hertz to Mels

	Args:
		freq: A value in Hertz. This can also be a numpy array.

	Returns
		A value in Mels.
	"""
	return 2595 * np.log10(1 + freq / 700.0)

def mel2freq(mel):
	"""Convert a value in Mels to Hertz

	Args:
		mel: A value in Mels. This can also be a numpy array.

	Returns
		A value in Hertz.
	"""
	return 700 * (10 ** (mel / 2595.0) - 1)

def compute_mfcc(signal, fs_hz, cmn=False):
	"""Compute MFCC features

	Args:
		signal: signal array
		fs_hz: sampling rate in Hz
		cmn: whether perform CMN

	Returns
		MFCC feature vectors (num_frames x num_ceps)
	"""

	signal_length = len(signal)

	# Define parameters
	preemph = 0.97				# pre-emphasis coeefficient
	frame_length_ms = 25		# frame length in ms
	frame_step_ms = 10			# frame shift in ms
	low_freq_hz = 0				# filterbank low frequency in Hz
	high_freq_hz = 8000			# filterbank high frequency in Hz
	nyquist = fs_hz / 2.0;		# Check the Nyquist frequency
	if high_freq_hz > nyquist:
		high_freq_hz = nyquist
	num_filters = 26			# number of mel-filters
	num_ceps = 12				# number of cepstral coefficients (excluding C0)
	cep_lifter = 22				# Cepstral liftering order
	eps = 0.001					# Floor to avoid log(0)

	# Pre-emphasis
	emphasised = np.append(signal[0], signal[1:] - preemph * signal[:-1]);

	# Compute number of frames and padding
	frame_length = int(round(frame_length_ms / 1000.0 * fs_hz));
	frame_step = int(round(frame_step_ms / 1000.0  * fs_hz));
	num_frames = int(np.ceil(float(signal_length - frame_length) / frame_step))

	pad_signal_length = num_frames * frame_step + frame_length
	pad_zeros = np.zeros((pad_signal_length - signal_length))
	pad_signal = np.append(emphasised, pad_zeros)

	# Find the smallest power of 2 greater than frame_length
	NFFT = 1<<(frame_length-1).bit_length(); 

	# Compute mel-filters
	mel_filters = np.zeros((NFFT // 2 + 1, num_filters))
	low_freq_mel = freq2mel(low_freq_hz)
	high_freq_mel = freq2mel(high_freq_hz)
	mel_bins = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2) # Equally spaced in the Mel scale
	freq_bins = mel2freq(mel_bins) # Convert Mel to Hz
	fft_bins = (NFFT + 1.0) * freq_bins // fs_hz # FFT bin indice for the filters
	for m in range(0, num_filters):
		for k in range(int(fft_bins[m]), int(fft_bins[m + 1])):
			mel_filters[k, m] = (k - fft_bins[m]) / (fft_bins[m+1] - fft_bins[m])
		for k in range(int(fft_bins[m + 1]), int(fft_bins[m + 2])):
			mel_filters[k, m] = (fft_bins[m + 2] - k) / (fft_bins[m + 2] - fft_bins[m+1])

	# Hamming window
	win = np.hamming(frame_length)

	# Lifter
	lift = 1 + (cep_lifter / 2.0) * np.sin(np.pi * np.arange(num_ceps) / cep_lifter)

	# Pre-allocation
	feat_mfcc = np.zeros((num_frames, num_ceps))

	# Compute MFCCs frame by frame
	for t in range(0, num_frames):

		# Framing
		frame = pad_signal[t*frame_step:t*frame_step+frame_length]

		# Apply the Hamming window
		frame = frame * win
		
		# Compute magnitude spectrum 
		magspec = np.absolute(np.fft.rfft(frame, NFFT))

		# Compute power spectrum
		powspec =  (magspec ** 2) * (1.0 / NFFT)

		# Compute log mel spectrum
		fbank = np.dot(powspec, mel_filters)
		fbank[fbank < eps] = eps # Avoid log(0)
		fbank = np.log(fbank)

		# Apply DCT to get num_ceps MFCCs, omit C0
		mfcc = dct(fbank, norm='ortho')[1:num_ceps+1] # Omit C0

		# Liftering
		mfcc *= lift 

		# Save mfcc features
		feat_mfcc[t, :] = mfcc

	# Cepstral mean and variance normalisation
	if cmn == True:
		feat_mfcc = (feat_mfcc - np.mean(feat_mfcc, axis=0)) / np.std(feat_mfcc, axis=0)
	return feat_mfcc;

