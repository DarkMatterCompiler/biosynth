import torch
from torch.cuda.amp import GradScaler, autocast
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.manifold import TSNE


def train_step(model, optimizer, criterion, fingerprint, iris, labels, device, scaler, accumulation_steps=4, step=0):
    model.train()
    optimizer.zero_grad()

    fingerprint = fingerprint.to(device)
    iris = iris.to(device)
    labels = labels.to(device)
    batch_size = fingerprint.size(0) // 2  # Adjust for instances_per_person

    with autocast():
        recon_fingerprint, recon_iris, fingerprint_emb, iris_emb, fused_emb = model(
            fingerprint, iris)
        loss, recon_loss, contrastive_loss = criterion(
            recon_fingerprint, fingerprint, recon_iris, iris, fingerprint_emb, iris_emb, fused_emb, labels)
        loss = loss / accumulation_steps  # Normalize loss

    scaler.scale(loss).backward()
    if (step + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss.item() * accumulation_steps, recon_loss, contrastive_loss


# In train.py, visualize_reconstructions
def visualize_reconstructions(model, dataset, device, num_samples=5, instances_per_person=2, epoch=0, save_dir='recon_plots'):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples * instances_per_person,
                             4, figsize=(12, 3 * num_samples * instances_per_person))
    if num_samples * instances_per_person == 1:
        axes = np.array([[axes]])

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            fingerprints = sample['fingerprints'].to(device)
            irises = sample['irises'].to(device)
            labels = sample['labels'].numpy()

            # Update unpacking to match model output
            recon_fingerprints, recon_irises, _, _, _ = model(
                fingerprints, irises)

            for j in range(instances_per_person):
                row = i * instances_per_person + j
                fingerprint = fingerprints[j].cpu().squeeze().numpy()
                iris = irises[j].cpu().squeeze().numpy()
                recon_fingerprint = recon_fingerprints[j].cpu(
                ).squeeze().numpy()
                recon_iris = recon_irises[j].cpu().squeeze().numpy()

                axes[row, 0].imshow(fingerprint, cmap='gray')
                axes[row, 0].set_title(f'Orig Fingerprint (P{labels[j]})')
                axes[row, 0].axis('off')

                axes[row, 1].imshow(recon_fingerprint, cmap='gray')
                axes[row, 1].set_title(f'Recon Fingerprint (P{labels[j]})')
                axes[row, 1].axis('off')

                axes[row, 2].imshow(iris, cmap='gray')
                axes[row, 2].set_title(f'Orig Iris (P{labels[j]})')
                axes[row, 2].axis('off')

                axes[row, 3].imshow(recon_iris, cmap='gray')
                axes[row, 3].set_title(f'Recon Iris (P{labels[j]})')
                axes[row, 3].axis('off')

    plt.suptitle(f'Reconstructions (Epoch {epoch+1})')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(
        save_dir, f'reconstructions_epoch_{epoch+1}.png'), bbox_inches='tight')
    plt.close()


def visualize_embeddings(model, dataset, device, num_persons=10, num_samples=4, epoch=0, save_dir='embedding_plots'):
    model.eval()
    embeddings = []
    labels = []
    np.random.seed(42)

    person_ids = dataset.person_ids[:min(num_persons, len(dataset.person_ids))]

    with torch.no_grad():
        for person_id in person_ids:
            for _ in range(num_samples):
                label = dataset.label_map[person_id]
                iris_img_path = np.random.choice(
                    dataset.iris_images[person_id])
                fingerprint_img_path = np.random.choice(
                    dataset.fingerprint_images[person_id])

                transform = dataset.transform
                iris_img = transform(Image.open(iris_img_path).convert(
                    'L')).unsqueeze(0).to(device)
                fingerprint_img = transform(Image.open(
                    fingerprint_img_path).convert('L')).unsqueeze(0).to(device)

                # Update unpacking to match model output
                _, _, _, _, emb = model(fingerprint_img, iris_img)
                embeddings.append(emb.cpu().numpy()[0])
                labels.append(label)
    # Rest of the function remains the same
    if len(embeddings) < 10:
        print(
            f"Warning: Only {len(embeddings)} samples, t-SNE may be unreliable.")
        return

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    intra_dist, inter_dist = compute_distances(
        torch.tensor(embeddings), torch.tensor(labels))
    print(
        f"Intra-class distance: {intra_dist:.4f}, Inter-class distance: {inter_dist:.4f}")
    plt.figure(figsize=(10, 8))
    for i in range(len(person_ids)):
        idx = [j for j, l in enumerate(labels) if l == i]
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    label=f'Person {person_ids[i]}', s=50, alpha=0.7)

    plt.title(f'Embedding Visualization (Epoch {epoch+1})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(
        save_dir, f'embeddings_epoch_{epoch+1}.png'), bbox_inches='tight')
    plt.close()


def compute_metrics(original, reconstructed):
    original = original.cpu().numpy().squeeze()
    reconstructed = reconstructed.cpu().numpy().squeeze()
    psnr = peak_signal_noise_ratio(original, reconstructed, data_range=2.0)
    ssim = structural_similarity(original, reconstructed, data_range=2.0)
    return psnr, ssim


def compute_distances(embeddings, labels):
    intra_dist = []
    inter_dist = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = torch.norm(embeddings[i] - embeddings[j]).item()
            if labels[i] == labels[j]:
                intra_dist.append(dist)
            else:
                inter_dist.append(dist)
    return np.mean(intra_dist) if intra_dist else 0, np.mean(inter_dist) if inter_dist else float('inf')
