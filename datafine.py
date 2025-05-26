import os
import random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import glob


class HazeDataset(Dataset):
    def __init__(self, ori_root, haze_root, transforms=None):
        """
        Dataset class for paired haze and reference images.
        Assumes both folders contain images with identical filenames (e.g., img001.jpg in both).
        """
        self.ori_root = ori_root
        self.haze_root = haze_root
        self.transforms = transforms

        self.matching_dict = {}      # Maps filename → (clean path, [hazy paths])
        self.ori_image_list = []     # List of clean image paths (each with at least 1 matching hazy)

        self.build_pair_dict()
        print("✅ Total matched image pairs:", len(self.ori_image_list))

    def build_pair_dict(self):
        haze_image_list = glob.glob(os.path.join(self.haze_root, '*.jpg')) + \
                          glob.glob(os.path.join(self.haze_root, '*.png'))

        for haze_path in haze_image_list:
            filename = os.path.basename(haze_path)
            clean_path = os.path.join(self.ori_root, filename)

            if os.path.exists(clean_path):
                if filename not in self.matching_dict:
                    self.matching_dict[filename] = {
                        "clean": clean_path,
                        "hazes": []
                    }
                    self.ori_image_list.append(filename)
                self.matching_dict[filename]["hazes"].append(haze_path)

        random.shuffle(self.ori_image_list)

    def __len__(self):
        return len(self.ori_image_list)

    def __getitem__(self, idx):
        filename = self.ori_image_list[idx]
        paths = self.matching_dict[filename]
        clean_path = paths["clean"]
        haze_path = random.choice(paths["hazes"])

        try:
            ori_image = Image.open(clean_path).convert("RGB")
            haze_image = Image.open(haze_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            print(f"❌ Skipping corrupted image at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transforms:
            ori_image = self.transforms(ori_image)
            haze_image = self.transforms(haze_image)

        return ori_image, haze_image


# 🔍 Optional: Run as a standalone script for debugging
if __name__ == "__main__":
    from torchvision import transforms

    ori_dir = "./data/train/reference"
    haze_dir = "./data/train/raw"

    transform_fn = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = HazeDataset(ori_dir, haze_dir, transforms=transform_fn)

    for i in range(5):
        ori_img, haze_img = dataset[i]
        print(f"[{i}] Shapes — Ori: {ori_img.shape}, Haze: {haze_img.shape}")
