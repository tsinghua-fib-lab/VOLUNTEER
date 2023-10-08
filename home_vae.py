import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Parameters
input_dim = 4
latent_dim = 2
batch_size = 64
epochs = 1000
lr = 1e-4

torch.manual_seed(42)
np.random.seed(42)


# Dataset
class CoordinateDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

# VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2*latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        return mu, logvar
        
    def decode(self, z):
        x = self.decoder(z)
        return x
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Train
def train_vae(vae, train_loader, optimizer, epoch):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        # loss
        recon_loss = nn.MSELoss(reduction='sum')(recon_batch, data)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch {} Train loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# Generate
def generate_data(vae, n_samples, scaler):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, vae.latent_dim)
        generated_coords = vae.decode(z)
        generated_coords_unscaled = scaler.inverse_transform(generated_coords.numpy())
        generated_df = pd.DataFrame(generated_coords_unscaled, columns=['home_latitude', 'home_longitude', 'work_latitude', 'work_longitude'])
        user_ids = np.arange(0, n_samples)
        generated_df['user_id'] = user_ids
        output_cols = ['user_id', 'home_latitude', 'home_longitude', 'work_latitude', 'work_longitude']
        generated_data = generated_df[output_cols].values.tolist()
        return generated_data


# Load
data = pd.read_csv('home_work_lat_lon.csv')

# Normalizing
scaler = StandardScaler()
coords = data[['home_latitude', 'home_longitude', 'work_latitude', 'work_longitude']].values
coords_scaled = scaler.fit_transform(coords)

tensor_data = torch.tensor(coords_scaled, dtype=torch.float32)


# Model
vae = VAE(input_dim, latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

# Dataset
train_data = CoordinateDataset(tensor_data)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Train
for epoch in range(1, epochs+1):
    train_vae(vae, train_loader, optimizer, epoch)

# Generate
n_samples = 6122
generated_data = generate_data(vae, n_samples, scaler)

# Save data
generated_df = pd.DataFrame(generated_data, columns=['user_id', 'home_latitude', 'home_longitude', 'work_latitude', 'work_longitude'])
generated_df.to_csv('generated_data.csv', index=False)

# Save model
torch.save(vae.state_dict(), 'vae_model.pt')



# Load model
#vae = VAE(input_dim=4, latent_dim=2)
#vae.load_state_dict(torch.load('vae_model.pt'))
#vae.eval()

# Generate
#n_samples = 2780
#generated_data = generate_data(vae, n_samples, scaler)

# Save data
#generated_df = pd.DataFrame(generated_data, columns=['user_id', 'home_latitude', 'home_longitude', 'work_latitude', 'work_longitude'])
#generated_df.to_csv('generated_data.csv', index=False)