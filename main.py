# from train import Trainer
# from torch.utils.data import DataLoader
# import torch
# from torchvision import transforms
# from dataset import CustomBiometricDataset
# # 👈 Make sure this path matches your file
# from multimodal_model import MultimodalEmbeddingNet

# # ---------- 🔧 Configuration ----------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image_size = (128, 128)
# batch_size = 8
# num_classes = 100  # 🔁 Change this to match your actual number of identities

# # ---------- 🧼 Transforms ----------
# transform_dict = {
#     'forehead': transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ConvertImageDtype(torch.float)
#     ]),
#     'iris': transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ConvertImageDtype(torch.float)
#     ]),
#     'periocular': transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ConvertImageDtype(torch.float)
#     ]),
# }

# # ---------- 📁 Dataset & Loader ----------
# dataset = CustomBiometricDataset(
#     data_dir='data',
#     transform=transform_dict,
#     sample_one_per_identity=True
# )

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # ---------- 🧠 Model ----------
# model = MultimodalEmbeddingNet(
#     embedding_dim=256,
#     num_classes=num_classes
# ).to(device)

# # ---------- 🔁 Forward Pass (Test) ----------
# batch = next(iter(dataloader))
# iris = batch['iris'].to(device)
# periocular = batch['periocular'].to(device)
# forehead = batch['forehead'].to(device)
# labels = batch['label'].to(device)

# embedding, logits = model(iris=iris, periocular=periocular, forehead=forehead)

# print("✅ Embedding shape:", embedding.shape)  # [B, 256]
# print("✅ Logits shape:", logits.shape)        # [B, num_classes]
# print("✅ Labels:", labels)

# config = {
#     'lr': 1e-4,
#     'weight_decay': 1e-5,
#     'num_classes': num_classes,  # 👈 Add this line
#     'device': device             # optional, used by CFPC_loss or others
# }
# from train import Trainer

# trainer = Trainer(model, device, config)

# for epoch in range(1, 11):  # Run 10 epochs
#     trainer.train_one_epoch(dataloader, epoch)

from tsne_plot import plot_tsne
from validate import validate, extract_embeddings
from train import Trainer
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from dataset import CustomBiometricDataset
from multimodal_model import MultimodalEmbeddingNet

# ---------- 🔧 Config ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = (128, 128)
batch_size = 8

# ---------- 🧼 Transforms ----------
transform_dict = {
    'forehead': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ConvertImageDtype(torch.float)
    ]),
    'iris': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ConvertImageDtype(torch.float)
    ]),
    'periocular': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ConvertImageDtype(torch.float)
    ]),
}

# ---------- 📁 Dataset ----------
dataset = CustomBiometricDataset(
    data_dir='data',
    transform=transform_dict,
    sample_one_per_identity=True
)

num_classes = len(dataset.person_ids)  # ✅ based on label remapping

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------- 🧠 Model ----------
model = MultimodalEmbeddingNet(
    embedding_dim=256,
    num_classes=num_classes
).to(device)

# ---------- ⚙️ Train Config ----------
config = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'num_classes': num_classes,
    'device': device
}

# ---------- 🚀 Training ----------
trainer = Trainer(model, device, config)

for epoch in range(1, 11):
    trainer.train_one_epoch(dataloader, epoch)

# After training
validate(model, dataloader, device)

# If you want t-SNE:
embs, labs = extract_embeddings(model, dataloader, device)
plot_tsne(embs.numpy(), labs.numpy())
