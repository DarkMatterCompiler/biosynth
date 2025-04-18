import torch
import torch.nn as nn


class FingerprintEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(FingerprintEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, latent_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc(x)))
        return x


class IrisEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(IrisEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, latent_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc(x)))
        return x


class FingerprintDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(FingerprintDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.upsample3 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.conv_out = nn.Conv2d(8, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(-1, 64, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.upsample1(x)
        x = self.relu(self.conv2(x))
        x = self.upsample2(x)
        x = self.relu(self.conv3(x))
        x = self.upsample3(x)
        x = self.sigmoid(self.conv_out(x))
        return x


class IrisDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(IrisDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.upsample3 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.conv_out = nn.Conv2d(8, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(-1, 64, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.upsample1(x)
        x = self.relu(self.conv2(x))
        x = self.upsample2(x)
        x = self.relu(self.conv3(x))
        x = self.upsample3(x)
        x = self.sigmoid(self.conv_out(x))
        return x


class FusionTransformer(nn.Module):
    def __init__(self, input_dim=512, latent_dim=256, nhead=4, dim_feedforward=1024, dropout=0.1, num_prompts=1):
        super(FusionTransformer, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.linear_in = nn.Linear(input_dim, latent_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                latent_dim, nhead, dim_feedforward, dropout),
            num_layers=2
        )
        self.norm = nn.LayerNorm(latent_dim)

        # MPT: Learnable prompt embeddings
        self.num_prompts = num_prompts
        self.prompt_emb = nn.Parameter(torch.randn(num_prompts, latent_dim))
        self.prompt_conv = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.prompt_relu = nn.ReLU()

    def forward(self, fingerprint_emb, iris_emb):
        # Concatenate fingerprint and iris embeddings
        x = torch.cat((fingerprint_emb, iris_emb), dim=-1)
        x = self.linear_in(x)  # Shape: [batch_size, latent_dim]

        # Prepare prompt embeddings
        batch_size = x.size(0)
        # Shape: [batch_size, num_prompts, latent_dim]
        prompt = self.prompt_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        # Process prompts through 1x1 conv and ReLU
        # Shape: [batch_size, latent_dim, num_prompts]
        prompt = prompt.permute(0, 2, 1)
        # Shape: [batch_size, latent_dim, num_prompts]
        prompt = self.prompt_conv(prompt)
        prompt = self.prompt_relu(prompt)
        # Shape: [batch_size, num_prompts, latent_dim]
        prompt = prompt.permute(0, 2, 1)

        # Concatenate input embeddings and prompts
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, latent_dim]
        # Shape: [batch_size, 1 + num_prompts, latent_dim]
        combined = torch.cat((x, prompt), dim=1)

        # Pass through transformer
        # Shape: [batch_size, 1 + num_prompts, latent_dim]
        x = self.transformer(combined) + combined

        # Extract the output corresponding to the input embedding (ignore prompt outputs)
        x = x[:, 0, :]  # Shape: [batch_size, latent_dim]
        x = self.norm(x)
        return x


class BiometricAutoencoder(nn.Module):
    def __init__(self, encoder_latent_dim=256, fused_latent_dim=256, num_persons=None):
        super(BiometricAutoencoder, self).__init__()
        self.fingerprint_encoder = FingerprintEncoder(encoder_latent_dim)
        self.iris_encoder = IrisEncoder(encoder_latent_dim)
        self.fusion_transformer = FusionTransformer(
            input_dim=2 * encoder_latent_dim, latent_dim=fused_latent_dim)
        self.fingerprint_decoder = FingerprintDecoder(fused_latent_dim)
        self.iris_decoder = IrisDecoder(fused_latent_dim)
        self.num_persons = num_persons
        if num_persons is None:
            raise ValueError("num_persons must be provided for classification")
        self.fingerprint_proj = nn.Linear(encoder_latent_dim, num_persons)
        self.iris_proj = nn.Linear(encoder_latent_dim, num_persons)

    # In models.py, BiometricAutoencoder.forward
    def forward(self, fingerprint, iris):
        fingerprint_emb = self.fingerprint_encoder(fingerprint)
        iris_emb = self.iris_encoder(iris)
        fused_emb = self.fusion_transformer(fingerprint_emb, iris_emb)
        recon_fingerprint = self.fingerprint_decoder(fused_emb)
        recon_iris = self.iris_decoder(fused_emb)
        fingerprint_emb_proj = self.fingerprint_proj(fingerprint_emb)
        iris_emb_proj = self.iris_proj(iris_emb)
        return recon_fingerprint, recon_iris, fingerprint_emb_proj, iris_emb_proj, fused_emb
