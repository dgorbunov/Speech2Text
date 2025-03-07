#!/usr/bin/env python
"""
A complete script to load a subset of LibriSpeech,
extract MFCC features from the audio files, perform
PCA for dimensionality reduction, and visualize the result.
"""

import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def extract_features(file_path, n_mfcc=13):
    """
    Load an audio file and extract its MFCC features.
    The features are averaged over time to obtain a fixed-size vector.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def load_librispeech_features(root_dir, file_limit=100, n_mfcc=13):
    """
    Recursively search for .flac audio files in the LibriSpeech directory,
    extract MFCC features from a subset of files, and optionally retrieve speaker IDs.
    """
    # Find all FLAC files in the provided root directory
    audio_files = glob.glob(os.path.join(root_dir, '**', '*.flac'), recursive=True)
    features = []
    labels = []
    
    for file in audio_files[:file_limit]:
        feat = extract_features(file, n_mfcc)
        features.append(feat)
        # Extract speaker ID from the file path.
        # Adjust this based on your specific LibriSpeech folder structure.
        # For example, if the path is .../LibriSpeech/train-clean-100/84/121123/84-121123-0000.flac,
        # this gets the speaker folder (e.g., "84").
        speaker_id = os.path.basename(os.path.dirname(os.path.dirname(file)))
        labels.append(speaker_id)
        
    return np.array(features), labels

def main():
    # Set the path to your LibriSpeech dataset directory.
    # Update 'path_to_librispeech' to the correct path on your system.
    librispeech_root = 'data/LibriSpeech'
    
    print("Loading features from LibriSpeech dataset...")
    features, labels = load_librispeech_features(librispeech_root, file_limit=100, n_mfcc=13)
    
    # Standardize the features since PCA is sensitive to feature scaling.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA to reduce dimensions to 2 for easy visualization.
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_scaled)
    
    # Output the explained variance ratio for the principal components.
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    
    # Create a scatter plot of the PCA-reduced features.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

# Plot PCA-reduced features
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], 
                      c=range(len(labels)), cmap='viridis', alpha=0.7)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA Visualization of LibriSpeech MFCC Features')


    plt.show()

if __name__ == "__main__":
    main()

