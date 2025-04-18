import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class BiometricDataset(Dataset):
    def __init__(self, root_dir, transform=None, instances_per_person=2, image_size=128, val_split=0.2):
        """
        Args:
            root_dir (str): Root directory containing 'iris' and 'fingerprint' folders.
            transform (callable, optional): Optional transform to be applied to images.
            instances_per_person (int): Number of instances to return per person (default: 2).
            image_size (int): Target size of images (e.g., 128 or 256, default: 128).
            val_split (float): Fraction of dataset for validation (0 to disable, default: 0.2).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.instances_per_person = instances_per_person
        self.image_size = image_size
        self.val_split = val_split

        # Get list of person IDs
        self.iris_dir = os.path.join(root_dir, 'iris')
        self.fingerprint_dir = os.path.join(root_dir, 'fingerprint')
        self.person_ids = [f for f in sorted(os.listdir(
            self.iris_dir)) if f != '.DS_Store' and os.path.isdir(os.path.join(self.iris_dir, f))]

        # Store image paths for each person
        self.iris_images = {}
        self.fingerprint_images = {}
        self.excluded_persons = []

        for person_id in self.person_ids:
            iris_person_dir = os.path.join(self.iris_dir, person_id)
            fingerprint_person_dir = os.path.join(
                self.fingerprint_dir, person_id)

            if not os.path.exists(iris_person_dir) or not os.path.exists(fingerprint_person_dir):
                self.excluded_persons.append(person_id)
                print(f"[!] Skipping {person_id}: Missing directory")
                continue

            self.iris_images[person_id] = [
                os.path.join(iris_person_dir, img)
                for img in sorted(os.listdir(iris_person_dir))
                if img.endswith(('.jpg', '.jpeg', '.png'))
            ]
            self.fingerprint_images[person_id] = [
                os.path.join(fingerprint_person_dir, img)
                for img in sorted(os.listdir(fingerprint_person_dir))
                if img.endswith(('.BMP', '.jpg', '.png', '.jpeg'))
            ]

            # Warn if image count is less than expected (10)
            if len(self.iris_images[person_id]) < 10:
                print(
                    f"[!] Warning: Person {person_id} has {len(self.iris_images[person_id])} iris images (expected 10)")
            if len(self.fingerprint_images[person_id]) < 10:
                print(
                    f"[!] Warning: Person {person_id} has {len(self.fingerprint_images[person_id])} fingerprint images (expected 10)")
            if len(self.iris_images[person_id]) == 0 or len(self.fingerprint_images[person_id]) == 0:
                self.excluded_persons.append(person_id)
                print(
                    f"[!] Skipping {person_id}: No images in {'iris' if len(self.iris_images[person_id]) == 0 else 'fingerprint'} directory")
                continue

        self.person_ids = [
            pid for pid in self.person_ids if pid not in self.excluded_persons]
        self.label_map = {pid: idx for idx, pid in enumerate(self.person_ids)}
        self.num_persons = len(self.person_ids)

        if self.num_persons == 0:
            print(
                "[!] Warning: No valid persons with complete image data found. Check your dataset.")
        else:
            print(
                f"[+] Found {self.num_persons} valid persons with complete image data.")

        # Split into train and validation if val_split > 0
        if self.val_split > 0 and 0 < self.val_split < 1 and self.num_persons > 0:
            self.train_person_ids, self.val_person_ids = self._split_dataset()
        else:
            self.train_person_ids = self.person_ids
            self.val_person_ids = []

        self.label_map = {pid: idx for idx, pid in enumerate(self.person_ids)}
        print("Label map:", self.label_map)
        print("Number of persons:", self.num_persons)
        if self.num_persons < 2:
            raise ValueError(
                "Dataset must have at least 2 persons for classification")

    def _split_dataset(self):
        """Split dataset into train and validation sets."""
        indices = list(range(self.num_persons))
        random.shuffle(indices)
        split_idx = int((1 - self.val_split) * self.num_persons)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        return [self.person_ids[i] for i in train_idx], [self.person_ids[i] for i in val_idx]

    def __len__(self):
        return len(self.train_person_ids) if self.val_split > 0 and self.num_persons > 0 else self.num_persons

    def __getitem__(self, idx):
        """
        Returns 'instances_per_person' instances for a person, each with a randomly selected iris and fingerprint.
        """
        if self.num_persons == 0:
            raise ValueError("No valid data available in the dataset.")

        person_id = self.train_person_ids[idx] if self.val_split > 0 else self.person_ids[idx]
        label = self.label_map[person_id]

        fingerprints = []
        irises = []
        labels = []

        # Generate instances
        for _ in range(self.instances_per_person):
            try:
                # Randomly select images
                iris_img_path = random.choice(self.iris_images[person_id])
                fingerprint_img_path = random.choice(
                    self.fingerprint_images[person_id])

                # Load images
                iris_img = Image.open(iris_img_path).convert('L')
                fingerprint_img = Image.open(fingerprint_img_path).convert('L')

                # Apply transforms
                if self.transform is not None:
                    iris_img = self.transform(iris_img)
                    fingerprint_img = self.transform(fingerprint_img)

                irises.append(iris_img)
                fingerprints.append(fingerprint_img)
                labels.append(label)
            except Exception as e:
                print(f"[!] Error loading images for {person_id}: {e}")
                # Fallback: Return zero tensors resized to target size
                irises.append(torch.zeros(
                    (1, self.image_size, self.image_size)))
                fingerprints.append(torch.zeros(
                    (1, self.image_size, self.image_size)))
                labels.append(label)

        return {
            # Shape: [instances_per_person, 1, image_size, image_size]
            'fingerprints': torch.stack(fingerprints),
            # Shape: [instances_per_person, 1, image_size, image_size]
            'irises': torch.stack(irises),
            # Shape: [instances_per_person]
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        labels.append(label)
        print(f"Person {person_id}: Label {label}")

    @property
    def val_dataset(self):
        """Return validation dataset if split was applied."""
        if not self.val_person_ids or self.num_persons == 0:
            return None
        return ValBiometricDataset(self, self.val_person_ids)


class ValBiometricDataset(Dataset):
    def __init__(self, parent_dataset, val_person_ids):
        self.parent = parent_dataset
        self.val_person_ids = val_person_ids

    def __len__(self):
        return len(self.val_person_ids)

    def __getitem__(self, idx):
        return self.parent.__getitem__(idx)


if __name__ == "__main__":
    # Example usage
    root_dir = './dataset'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = BiometricDataset(root_dir, transform=transform,
                               instances_per_person=2, image_size=128, val_split=0.2)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Total persons: {len(dataset)}")
    print(f"Validation persons: {len(dataset.val_dataset or [])}")
    print(f"Excluded persons: {dataset.excluded_persons}")

    # Visualize a sample if data exists
    if len(dataset) > 0:
        for i, batch in enumerate(dataloader):
            if i == 0:
                fingerprints = batch['fingerprints'].numpy()
                irises = batch['irises'].numpy()
                labels = batch['labels'].numpy()
                fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                axes[0, 0].imshow(fingerprints[0, 0, 0], cmap='gray')
                axes[0, 0].set_title(f"Fingerprint (Person {labels[0]})")
                axes[0, 1].imshow(irises[0, 0, 0], cmap='gray')
                axes[0, 1].set_title(f"Iris (Person {labels[0]})")
                axes[1, 0].imshow(fingerprints[1, 0, 0], cmap='gray')
                axes[1, 0].set_title(f"Fingerprint (Person {labels[1]})")
                axes[1, 1].imshow(irises[1, 0, 0], cmap='gray')
                axes[1, 1].set_title(f"Iris (Person {labels[1]})")
                plt.show()
                break
    else:
        print("[!] No valid data to visualize.")
