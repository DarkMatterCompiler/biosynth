import torch
import torch.nn as nn
from torchvision.models import vgg16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, recon, original):
        recon = recon.repeat(1, 3, 1, 1)
        original = original.repeat(1, 3, 1, 1)
        return self.mse(self.vgg(recon), self.vgg(original))


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    # In InfoNCELoss.forward
    def forward(self, embeddings, labels):
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        batch_size = embeddings.size(0)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)
        neg_mask = 1 - pos_mask

        # Select hard negatives (top-k similar negatives per sample)
        top_k = min(5, batch_size - 1)  # Avoid exceeding available negatives
        neg_values, _ = torch.topk(sim_matrix * neg_mask, k=top_k, dim=1)
        hard_neg_sim = neg_values.mean(dim=1)

        pos_sim = (sim_matrix * pos_mask).sum(dim=1)
        neg_sim = torch.logsumexp(sim_matrix * neg_mask, dim=1)
        loss = -torch.log(pos_sim / (hard_neg_sim + neg_sim + 1e-8) + 1e-8)
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, recon_weight=0.5, contrastive_weight=1.0, lm_weight=0.3, robustness_weight=0.2, num_persons=None):
        super(CombinedLoss, self).__init__()
        self.recon_loss = nn.MSELoss()
        self.contrastive_loss = InfoNCELoss(temperature=0.1)
        self.lm_loss = nn.CrossEntropyLoss()
        self.robustness_loss = nn.MSELoss()
        self.recon_weight = recon_weight
        self.contrastive_weight = contrastive_weight
        self.lm_weight = lm_weight
        self.robustness_weight = robustness_weight
        self.num_persons = num_persons

    def forward(self, recon_fingerprint, fingerprint, recon_iris, iris, fingerprint_emb, iris_emb, embeddings, labels):

        assert labels.min() >= 0, "Negative labels detected"
        assert self.num_persons is not None, "num_persons not provided"
        assert labels.max(
        ) < self.num_persons, f"Labels exceed num_persons ({self.num_persons})"

        fingerprint_recon_loss = self.recon_loss(
            recon_fingerprint, fingerprint)
        iris_recon_loss = self.recon_loss(recon_iris, iris)
        recon_loss = fingerprint_recon_loss + iris_recon_loss

        contrastive_loss = self.contrastive_loss(embeddings, labels)

        lm_fingerprint_loss = self.lm_loss(fingerprint_emb, labels)
        lm_iris_loss = self.lm_loss(iris_emb, labels)
        lm_loss = lm_fingerprint_loss + lm_iris_loss

        perturbed_embeddings = embeddings + torch.randn_like(embeddings) * 0.1
        robustness_loss = self.robustness_loss(
            perturbed_embeddings, embeddings)

        return (self.recon_weight * recon_loss +
                self.contrastive_weight * contrastive_loss +
                self.lm_weight * lm_loss +
                self.robustness_weight * robustness_loss), recon_loss.item(), contrastive_loss.item()
