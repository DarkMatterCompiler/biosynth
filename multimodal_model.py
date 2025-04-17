import torch
import torch.nn as nn
import torchvision.models as models


from torchvision.models import resnet18, ResNet18_Weights


class ModalityBranch(nn.Module):
    def __init__(self, output_dim=256, use_pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        base_model = resnet18(weights=weights)

        self.backbone = nn.Sequential(
            *list(base_model.children())[:-1])  # Remove final FC
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.backbone(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class MultimodalEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=None, use_metadata=False, metadata_dim=0):
        super().__init__()
        self.use_metadata = use_metadata

        # One ResNet branch per modality
        self.iris_branch = ModalityBranch()
        self.periocular_branch = ModalityBranch()
        self.forehead_branch = ModalityBranch()

        input_dim = 3 * 256  # Concatenated features from each branch
        if use_metadata:
            input_dim += metadata_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)  # Final shared identity embedding
        )

        self.classifier = nn.Linear(
            embedding_dim, num_classes) if num_classes else None

    def forward(self, iris, periocular, forehead, metadata=None):
        # Individual modality embeddings
        emb_iris = self.iris_branch(iris)         # modality "a"
        emb_periocular = self.periocular_branch(periocular)  # modality "p"
        emb_forehead = self.forehead_branch(forehead)        # modality "f"

        # Classification logits (optional heads)
        logits_f = self.classifier(emb_forehead) if self.classifier else None
        logits_p = self.classifier(emb_periocular) if self.classifier else None

        return logits_f, logits_p, emb_forehead, emb_periocular, emb_iris
