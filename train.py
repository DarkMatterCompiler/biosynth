import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from loss import TotalLoss


class Trainer:
    def __init__(self, model, device, config):
        self.device = device
        self.model = model.to(device)

        # Loss function: total LM + contrastive
        self.criterion = TotalLoss(
            num_classes=config['num_classes'], device=device).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        self.scaler = GradScaler()  # Mixed precision support

    def train_one_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            iris = batch['iris'].to(self.device)
            periocular = batch['periocular'].to(self.device)
            forehead = batch['forehead'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                # 🔄 Expected model output: all required parts for TotalLoss
                logits_f, logits_p, emb_f, emb_p, emb_a = self.model(
                    iris=iris,
                    periocular=periocular,
                    forehead=forehead
                )

                loss = self.criterion(
                    logits_f, logits_p,
                    emb_f, emb_p, emb_a,
                    labels
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"🔁 Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        return avg_loss
