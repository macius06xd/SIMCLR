from itertools import combinations

import h5py
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from tensorflow.python.keras import models
from torchvision import transforms
from PIL import Image
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
import os

import main

# Load SimCLR model and set it to evaluation mode
# Replace 'YourSimCLRModel' and 'checkpoint_epoch_17.pt' with your model and checkpoint names
# Example: model = YourSimCLRModel()
base_model = models.resnet101(pretrained = True)
model = main.SimCLRModel(base_model,128)
checkpoint = torch.load('checkpoints/checkpoint_epoch_99.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()





transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Extract features and labels
image_features = []
labels = []
image_paths_combined = []

# Create an HDF5 file
hdf5_file = 'image_features.hdf'
with h5py.File(hdf5_file, 'w') as hf:
    filenames = []  # To maintain order of filenames
    data_dir = 'data'  # Path to your data directory
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            image_paths = [os.path.join(label_path, img_file) for img_file in os.listdir(label_path)]
            image_paths_combined.extend(image_paths)
            # Extract features for all images in the folder
            for i in range(len(image_paths)):
                    img1 = Image.open(image_paths[i]).convert('RGB')
                    img2 = Image.open(image_paths[i]).convert('RGB')

                    # Apply transformations to convert images to tensors
                    img1 = transform(img1)
                    img2 = transform(img2)

                    # Forward pass through the model to get z1 and z2
                    with torch.no_grad():
                        _, _, z1, z2 = model(img1.unsqueeze(0), img2.unsqueeze(0))  # Add batch dimension

                    # Concatenate z1 and z2 features
                    combined_feature = z1.squeeze().cpu().numpy()
                    # Get the filename without extension
                    filename = os.path.basename(image_paths[i])
                    # Store the filename without extension
                    filenames.append(filename)
                    # Save the image feature as a dataset
                    hf.create_dataset(filename, data=combined_feature)
                    # Append features and labels
                    image_features.append(combined_feature)
                    labels.append(label)
    # Store filenames as a dataset
    hf.create_dataset('filenames', data=np.array(filenames, dtype='S'))
# Convert lists to numpy arrays
image_features = np.array(image_features)
labels = np.array(labels)

# Perform clustering on the features (you might need to tune parameters)
n_clusters = len(np.unique(labels))  # Number of clusters based on the ground truth labels
kmeans = KMeans(n_clusters=n_clusters,max_iter=1000,n_init=50)
cluster_preds = kmeans.fit_predict(image_features)
import os
import shutil

# Create directories for each cluster
result_dir = 'result'
for cluster_num in range(n_clusters):
    cluster_dir = os.path.join(result_dir, f'cluster_{cluster_num}')
    os.makedirs(cluster_dir, exist_ok=True)

# Copy images to their respective cluster folders
for i, img_path in enumerate(image_paths_combined):
    img_label = labels[i]
    img_cluster = cluster_preds[i]

    # Get destination directory based on cluster number
    dest_dir = os.path.join(result_dir, f'cluster_{img_cluster}')

    # Create a unique file name for the copied image (if needed)
    filename = os.path.basename(img_path)
    dest_filename = f'{filename}'  # You might want to modify this naming scheme

    # Copy the image to the destination directory
    shutil.copy(img_path, os.path.join(dest_dir, dest_filename))
# Calculate V-measure
v_measure = v_measure_score(labels, cluster_preds)
print("V-Measure Score:", v_measure)

