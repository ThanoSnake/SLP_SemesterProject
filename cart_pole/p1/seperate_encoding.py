import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
# Define VAE class with customizable latent size
class VAE(nn.Module):

    ###  torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)  ###
    # This network takes the 3-channel image and creates 16->32->64 channels         #
    # Each layer halves the dimensions (80/8 = 10, 120/8 = 15) * 64 channels         #
    # Finally it encodes it into the latent space of [latent_size] (mean, logvar)    #
    #                                                                                #
    # Then the decoder reverses this process                                         #
    def __init__(self, latent_size=128):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_logvar = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_decode = nn.Linear(latent_size, 64 * 10 * 15)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    ### This function returns a sample from a distribution defined by mean mu and std logvar ###
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    ###                                    Forward pass                                   ###
    # Input -> 3 conv layers -> flatten for mu, logvar -> Random sample -> decode -> output #
    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x).view(batch_size, -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.fc_decode(z).view(batch_size, 64, 10, 15)
        reconstructed = self.decoder(decoded)
        return reconstructed, mu, logvar

# Define loss function
### We have 2 separate losses (ONLY for dimensions 4-63)                                ###
# Reconstruction loss (MSE): Matches output image to input image                          #
# KL divergence: Regularizes latent space to be close to standard normal distribution     #
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# Training function
### We add an ADDITIONAL loss on dimensions 0-3 to match                         ###
# the true state variables (position, velocity, angle, angular velocity)           #
def train_vae(model, train_loader, optimizer, device,epoch):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        input, label, action, use1, use2 = data
        x = input.to(device)
        optimizer.zero_grad()
        use1=use1.to(device)
        recon_x, mu, logvar = model(x)
        loss1 = vae_loss(recon_x, x, mu[:,4:], logvar[:,4:])
        mu_label_loss = F.mse_loss(mu[:,0:4], use1)
        loss = loss1 +  mu_label_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader.dataset)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--train', type=str,default="./cartpole/train", help='Path to input file')
    parser.add_argument('--test', type=str,default="./cartpole/test", help='Path to input file')
    parser.add_argument('--save', type=str, default='./',
                        help='Path to output dir ')

    args = parser.parse_args()
    from loader import VaeDataset, SequenceDataset

    trainpath=args.train
    batch_size=32
    traindataset = VaeDataset(root=trainpath,shift=args.weight)
    TrainLoader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, drop_last=True)
    testdataset = VaeDataset(root=args.test,shift=args.weight)
    TestLoader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop with custom latent size
    latent_size = 64  # Set your desired latent size here
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    vae_model = VAE(latent_size=latent_size).to(device)
    optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

    epochs = 200
    for epoch in range(epochs):
        train_loss = train_vae(vae_model, TrainLoader, optimizer, device,epoch)
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}')

    # Save model
        torch.save(vae_model.state_dict(), 'vae_separate.pth')
