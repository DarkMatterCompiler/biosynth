# validate.py
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            iris = batch['iris'].to(device)
            periocular = batch['periocular'].to(device)
            forehead = batch['forehead'].to(device)
            label = batch['label'].to(device)

            _, _, emb_f, emb_p, emb_a = model(
                iris=iris, periocular=periocular, forehead=forehead)

            # You can average modalities, or use only one
            fused = (emb_f + emb_p + emb_a) / 3.0
            embeddings.append(fused.cpu())
            labels.append(label.cpu())

    return torch.cat(embeddings), torch.cat(labels)


def cosine_match(embeddings, labels):
    sims = cosine_similarity(embeddings)
    correct = 0
    total = len(embeddings)

    for i in range(total):
        sims[i, i] = -1  # ignore self
        best_match = sims[i].argmax()
        if labels[i] == labels[best_match]:
            correct += 1

    accuracy = correct / total
    return accuracy


def validate(model, dataloader, device):
    embeddings, labels = extract_embeddings(model, dataloader, device)
    accuracy = cosine_match(embeddings.numpy(), labels.numpy())
    print(f"🔍 Cosine Verification Accuracy: {accuracy:.4f}")
