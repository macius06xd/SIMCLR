import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
from PIL import Image
from gaussian import GaussianBlur
from efficientnet_pytorch import EfficientNet
import tracemalloc
tracemalloc.start()
# Define the DataPoint class
class DataPoint:
    def __init__(self, path, real_class):
        self.path = path
        self.real_class = real_class
        self.image = Image.open(path)


# Load and augment data function
def load_data_and_augment(data_dir):
    data_points = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]

            for path in image_paths:
                data_point = DataPoint(path, class_name)
                data_points.append(data_point)

    return data_points


# Data preprocessing

s = 1
size = 256
color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomApply([color_jitter], p=0.8),
                                      transforms.RandomGrayscale(p=0.2),
                                      GaussianBlur(kernel_size=int(0.1 * size)),
                                      transforms.ToTensor()
                                      ])

transform1 = data_transforms
transform2 = data_transforms
checkpoint_dir = "checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading and augmentation
class SimCLRDataset(Dataset):
    def __init__(self, data_points, transform1, transform2):
        self.data_points = data_points
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        image = self.data_points[idx].image
        img1 = self.transform1(image)
        img2 = self.transform2(image)
        return img1, img2


# Define the SimCLR model
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=64):
        super(SimCLRModel, self).__init__()
        self.encoder = base_encoder
        self.projector = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)  # Assuming x1 is a 3D tensor, adding batch dimension
        h2 = self.encoder(x2)  # Assuming x2 is a 3D tensor, adding batch dimension
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return h1, h2, z1, z2


# SimCLR Loss
class SimCLRLoss(nn.Module):
    def __init__(self, device, temperature=0.5, ):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, z1, z2):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(logits.shape[0]).to(logits.device)
        loss = self.criterion(logits, labels)
        return loss


# Hyperparameters search space
batch_sizes = [32, 64, 128]
learning_rates = [1e-4, 1e-3, 1e-2]
projection_dims = [64, 128, 256]
temperatures = [0.1, 0.5, 1.0]
batch_sizes = [32]
learning_rates = [1e-5]
projection_dims = [100]
temperatures = [0.4]
if __name__ == '__main__':

    # Training loop with hyperparameter search
    session_num = 0
    # Load and augment data
    data_dir = 'Data\\'
    data_points = load_data_and_augment(data_dir)

    # Create SimCLRDataset

    simclr_dataset = SimCLRDataset(data_points, transform1, transform2)
    simclr_dataloader = DataLoader(simclr_dataset, batch_size=128, shuffle=True, num_workers=0)
    for learning_rate in learning_rates:
        for projection_dim in projection_dims:
            for temperature in temperatures:
                # Set hyperparameters
                hparams = {
                    'learning_rate': learning_rate,
                    'projection_dim': projection_dim,
                    'temperature': temperature,
                    'num_epochs': 100
                }

                # Initialize ResNet-34 as the base encoder
                # Load EfficientNet (you may need to install efficientnet_pytorch package)
                base_model = models.resnet101(pretrained = True)
                # Set requires_grad to True for all parameters
                # Load MobileNetV3
                for param in base_model.parameters():
                    param.requires_grad = False

                # Initialize model, loss, and optimizer
                simclr_model = SimCLRModel(base_model, projection_dim=hparams['projection_dim']).to(device)
                simclr_loss = SimCLRLoss(device, temperature=hparams['temperature']).to(device)
                optimizer = optim.Adam(simclr_model.parameters(), lr=hparams['learning_rate'])

                # TensorBoard setup
                log_dir = f"logs/hparam_tuning/run-{session_num}"
                writer = SummaryWriter(log_dir)

                # Training loop
                loss_values = []  # Create an empty list to store loss values

                for epoch in range(hparams['num_epochs']):
                    for data in simclr_dataloader:
                        images1, images2 = data[0].to(device), data[1].to(device)
                        # Forward pass
                        h1, h2, z1, z2 = simclr_model(images1, images2)
                        loss = simclr_loss(z1, z2)

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Store the loss value
                        loss_values.append(loss.item())

                    # Log loss values to TensorBoard after each epoch
                    avg_loss = sum(loss_values) / len(loss_values)
                    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
                    loss_values = []  # Clear the list for the next epoch
                    # Save model checkpoint
                    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': simclr_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_filename)
                    print(avg_loss)
                    print("Epoch koniec")
                    # Log hyperparameters to TensorBoard
                    with SummaryWriter(log_dir, "hparams") as writer_hparams:
                        hparams_dict = {param_name: param_value for param_name, param_value in hparams.items()}
                        writer_hparams.add_hparams(hparams_dict, {'Loss/train_epoch': avg_loss})
