import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
from PIL import Image
from gaussian import GaussianBlur
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

# Logging support - tensorboard and hparams used for logging results and hyperparameters
# Saving/Checkpointing support - checkpoints are saved during training
# Hyperparameters selection logging - hyperparameters are logged with tensorboard hparams
class DataPoint:
    def __init__(self, path, real_class):
        self.path = path
        self.real_class = real_class

# Building the model - the architecture and loss function are defined in the SimCLRModel and SimCLRLoss classes
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

s=1
size = 256
color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])

# Logs from a sanity check - the training loop includes logging of loss which can be used for a sanity check
transform1 = data_transforms
transform2 = data_transforms
checkpoint_dir = "checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SimCLRDataset(Dataset):
    def __init__(self, data_points, transform1, transform2):
        self.data_points = data_points
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        image = Image.open(self.data_points[idx].path)
        img1 = self.transform1(image)
        img2 = self.transform2(image)
        return img1, img2

class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_encoder
        self.projector = nn.Sequential(
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Linear(512, projection_dim) 
        )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return h1, h2, z1, z2

class SimCLRLoss(nn.Module):
    def __init__(self,device, temperature=0.5,):
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

batch_sizes = [32, 64, 128]
learning_rates = [1e-4, 1e-3, 1e-2]
projection_dims = [64, 128, 256]
temperatures = [0.1, 0.5, 1.0]
batch_sizes = [32]
learning_rates = [1e-4]
projection_dims = [64]
temperatures = [0.1]
if __name__ == '__main__':

    session_num = 0
    data_dir = 'Data\\'
    data_points = load_data_and_augment(data_dir)

    simclr_dataset = SimCLRDataset(data_points, transform1, transform2)
    simclr_dataloader = DataLoader(simclr_dataset, batch_size=32, shuffle=True, num_workers=4)
    for learning_rate in learning_rates:
            for projection_dim in projection_dims:
                for temperature in temperatures:

                    hparams = {
                        'learning_rate': learning_rate,
                        'projection_dim': projection_dim,
                        'temperature': temperature,
                        'num_epochs': 5
                    }

                    base_encoder = models.resnet34(pretrained=True)

                    for param in base_encoder.parameters():
                        param.requires_grad = False

                    simclr_model = SimCLRModel(base_encoder, projection_dim=hparams['projection_dim']).to(device)
                    simclr_loss = SimCLRLoss(device,temperature=hparams['temperature']).to(device)
                    optimizer = optim.Adam(simclr_model.parameters(), lr=hparams['learning_rate'])

                    log_dir = f"logs/hparam_tuning/run-{session_num}"
                    writer = SummaryWriter(log_dir)

                    loss_values = []

                    for epoch in range(hparams['num_epochs']):
                        for data in simclr_dataloader:
                            images1, images2 = data[0].to(device), data[1].to(device)

                            h1, h2, z1, z2 = simclr_model(images1, images2)
                            loss = simclr_loss(z1, z2)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            loss_values.append(loss.item())

                        avg_loss = sum(loss_values) / len(loss_values)
                        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
                        loss_values = [] 

                        # Saving checkpoints for each epoch
                        checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': simclr_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss,
                        }, checkpoint_filename)

                        # Logging hyperparameters and metrics
                        with SummaryWriter(log_dir, "hparams") as writer_hparams:
                            hparams_dict = {param_name: param_value for param_name, param_value in hparams.items()}
                            writer_hparams.add_hparams(hparams_dict, {'Loss/train_epoch': avg_loss})

    # extracting vectors of photos
    def extract_features(simclr_model, data_loader):
        simclr_model.eval()
        features = {}
        with torch.no_grad():
            for data in data_loader:
                images1, images2 = data
                images1, images2 = images1.to(device), images2.to(device)
                h1, h2, z1, z2 = simclr_model(images1, images2)  
                
                for batch_index in range(len(images1)):
                    img_name = os.path.basename(data_loader.dataset.data_points[batch_index].path)
                    features[img_name] = z1[batch_index].cpu().numpy() 

        return features
    
    def extract_features(simclr_model, data_loader):
        simclr_model.eval() 
        features = {}
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                images1, images2 = data
                images1, images2 = images1.to(device), images2.to(device)
                h1, h2, z1, z2 = simclr_model(images1, images2) 
                batch_size = images1.size(0) 
                for batch_index in range(batch_size):
                    global_index = i * batch_size + batch_index
                    img_name = os.path.basename(data_loader.dataset.data_points[global_index].path)
                    features[img_name] = z1[batch_index].cpu().numpy() 
        return features


    feature_loader = DataLoader(simclr_dataset, batch_size=32, shuffle=False, num_workers=4)
    extracted_features = extract_features(simclr_model, feature_loader)

    import json
    with open('extracted_features.json', 'w') as f:
        json.dump({k: v.tolist() for k, v in extracted_features.items()}, f)

    for img_name, feature_vector in extracted_features.items():
        print(f"Obraz: {img_name}")
        print(f"Wektor cech: {feature_vector}\n")


    # k-means and v-measure
    n_clusters = 4

    feature_vectors = list(extracted_features.values())
    feature_labels = [data_point.real_class for data_point in simclr_dataset.data_points]

    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(feature_vectors)

    #print("feature labels:", len(feature_labels))
    #print("cluster labels:", len(cluster_labels))

    homogeneity = homogeneity_score(feature_labels, cluster_labels)
    completeness = completeness_score(feature_labels, cluster_labels)

    v_measure = v_measure_score(feature_labels, cluster_labels)

    print(f"Homogeneity: {homogeneity}")
    print(f"Completeness: {completeness}")
    print(f"V-measure: {v_measure}")






