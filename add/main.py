import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler
from dataset import BiometricDataset
from models import BiometricAutoencoder
from loss import CombinedLoss
from train import train_step, visualize_reconstructions, visualize_embeddings

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(
            f"Using GPU: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB memory")
        torch.cuda.empty_cache()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Dataset and DataLoader
    dataset = BiometricDataset(root_dir='./dataset', transform=transform,
                               instances_per_person=4, image_size=128, val_split=0.2)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_dataset = dataset.val_dataset
    val_loader = DataLoader(val_dataset, batch_size=8,
                            shuffle=False) if val_dataset else None

    # Model, criterion, optimizer, scheduler
    # In main.py, update model and criterion initialization
    model = BiometricAutoencoder(
        encoder_latent_dim=256,
        fused_latent_dim=256,
        num_persons=dataset.num_persons
    ).to(device)

    criterion = CombinedLoss(
        recon_weight=0.5,
        contrastive_weight=1.0,
        lm_weight=0.3,
        robustness_weight=0.2,
        num_persons=dataset.num_persons
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    warmup_epochs = 5
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs))
    scaler = GradScaler()
    accumulation_steps = 4

    # In main.py, modify the model and criterion initialization

    num_epochs = 20
    # Modify the training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_contrastive_loss = 0
        step = 0
        for batch_idx, batch in enumerate(train_loader):
            fingerprints = batch['fingerprints']
            irises = batch['irises']
            labels = batch['labels']

            batch_size, instances_per_person, _, h, w = fingerprints.shape
            fingerprints = fingerprints.view(
                batch_size * instances_per_person, 1, h, w).to(device)
            irises = irises.view(
                batch_size * instances_per_person, 1, h, w).to(device)
            labels = labels.view(batch_size * instances_per_person).to(device)

            # Update train_step call
            loss, recon_loss, contrastive_loss = train_step(
                model, optimizer, criterion, fingerprints, irises, labels, device, scaler, accumulation_steps, step)
            total_loss += loss
            total_recon_loss += recon_loss
            total_contrastive_loss += contrastive_loss
            step += 1

        # Validation loop
        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    fingerprints = batch['fingerprints']
                    irises = batch['irises']
                    labels = batch['labels']

                    batch_size, instances_per_person, _, h, w = fingerprints.shape
                    fingerprints = fingerprints.view(
                        batch_size * instances_per_person, 1, h, w).to(device)
                    irises = irises.view(
                        batch_size * instances_per_person, 1, h, w).to(device)
                    labels = labels.view(
                        batch_size * instances_per_person).to(device)

                    recon_fingerprint, recon_iris, fingerprint_emb, iris_emb, fused_emb = model(
                        fingerprints, irises)
                    loss, _, _ = criterion(recon_fingerprint, fingerprints, recon_iris,
                                           irises, fingerprint_emb, iris_emb, fused_emb, labels)
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = float('inf')

    # Rest of the code remains the same

        warmup_scheduler.step()
        scheduler.step()

        # Visualize reconstructions and embeddings
        visualize_reconstructions(model, dataset, device, num_samples=5,
                                  instances_per_person=dataset.instances_per_person, epoch=epoch)
        visualize_embeddings(model, dataset, device,
                             num_persons=10, num_samples=4, epoch=epoch)

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f} (Recon: {total_recon_loss:.4f}, Contrastive: {total_contrastive_loss:.4f}), Val Loss: {avg_val_loss:.4f}")
