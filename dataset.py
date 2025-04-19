import os
import random
from torchvision.io import read_image
from torch.utils.data import Dataset


class CustomBiometricDataset(Dataset):
    def __init__(self, data_dir, transform=None, sample_one_per_identity=True):
        self.data_dir = data_dir
        self.transform = transform or {}
        self.sample_one = sample_one_per_identity
        self.modalities = ['forehead', 'iris', 'periocular']

        self.person_ids = sorted(os.listdir(
            os.path.join(data_dir, 'forehead')))
        self.person_ids = [pid for pid in self.person_ids if os.path.isdir(
            os.path.join(data_dir, 'iris', pid))]
        self.id_to_label = {pid: idx for idx, pid in enumerate(
            self.person_ids)}  # 🔁 map to 0-based labels

        self.data = []
        for pid in self.person_ids:
            padded_pid = pid.zfill(3)
            person_data = {'id': pid}
            missing = False

            for modality in self.modalities:
                folder_pid = padded_pid if modality in [
                    'iris', 'periocular'] else pid
                folder = os.path.join(data_dir, modality, folder_pid)
                if not os.path.isdir(folder):
                    print(f"❌ Skipping {pid}: missing {modality} folder")
                    missing = True
                    break

                imgs = [f for f in os.listdir(folder) if f.lower().endswith(
                    ('.jpg', '.png', '.jpeg'))]
                if not imgs:
                    print(f"⚠️ Skipping {pid}: {modality} folder is empty")
                    missing = True
                    break

                person_data[modality] = [
                    os.path.join(folder, img) for img in imgs]

            if not missing:
                self.data.append(person_data)

        print(f"✅ Final dataset size: {len(self.data)} usable IDs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        person = self.data[idx]
        sample = {}

        for modality in self.modalities:
            img_list = person[modality]
            img_path = random.choice(
                img_list) if self.sample_one else img_list[0]
            img = read_image(img_path)
            if modality in self.transform:
                img = self.transform[modality](img)
            sample[modality] = img

        sample['label'] = self.id_to_label[person['id']]  # ✅ 0-based label
        return sample
