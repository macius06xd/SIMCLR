import numpy as np
import torch
from tensorflow.python.keras import models
from PIL import Image
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
import os
import main

# Define the function to extract features and perform clustering
def extract_features_and_cluster(checkpoint_file,images,labels):
    # Load SimCLR model and set it to evaluation mode
    base_encoder = models.resnet50(pretrained=True)
    model = main.SimCLRModel(base_encoder, 128)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model builded")
    image_features = []
    for i in images:
        # Forward pass through the model to get z1 and z2
        with torch.no_grad():
            _, _, z1, z2 = model(i.unsqueeze(0), i.unsqueeze(0))  # Add batch dimension

        # Concatenate z1 and z2 features
        combined_feature = z1.squeeze().cpu().numpy()
        image_features.append(combined_feature)

    # Perform clustering on the features (you might need to tune parameters)
    n_clusters = len(np.unique(labels))  # Number of clusters based on the ground truth labels
    print("Kmeans start")
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_preds = kmeans.fit_predict(image_features)

    # Create directories for each cluster
    # Calculate V-measure
    v_measure = v_measure_score(labels, cluster_preds)
    return v_measure
# Iterate through all 100 checkpoints
checkpoint_dir = 'checkpoints/'
checkpoint_files = [os.path.join(checkpoint_dir, f'checkpoint_epoch_{i+1}.pt') for i in range(100)]
# Extract features and labels
image_features = []
labels = []
image_paths_combined = []
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust size as needed
    transforms.ToTensor(),
])
images = []
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

            # Apply transformations to convert images to tensors
            img1 = transform(img1)
            img2 = img1
            filename = os.path.basename(image_paths[i])
            # Store the filename without extension
            filenames.append(filename)
            # Save the image feature as a dataset
            # Append features and labels
            labels.append(label)
            images.append(img1)
# Convert lists to numpy arrays
labels = np.array(labels)
print("zaczynamy")
for checkpoint_file in checkpoint_files:
    v_measure_score = extract_features_and_cluster(checkpoint_file,images,labels)
    print(f"For {checkpoint_file}: V-Measure Score = {v_measure_score}")
