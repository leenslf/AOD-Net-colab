import os
import torch
import random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import glob


class HazeDataset(Dataset):
    def __init__(self, ori_root, haze_root, transforms=None):
        self.ori_root = ori_root
        self.haze_root = haze_root
        self.transforms = transforms

        self.matching_dict = {}      # Maps original file name → [hazy variants]
        self.ori_image_list = []     # List of original image full paths

        self.build_pair_dict()
        print("✅ Total unique original images:", len(self.ori_image_list))

    def build_pair_dict(self):
        haze_image_list = glob.glob(os.path.join(self.haze_root, '*.jpg'))

        for haze_path in haze_image_list:
            filename = os.path.basename(haze_path)
            key = "_".join(filename.split("_")[:2]) + ".jpg"  # e.g., NYU2_999.jpg
            clean_path = os.path.join(self.ori_root, key)

            if os.path.exists(clean_path):
                if key not in self.matching_dict:
                    self.matching_dict[key] = []
                    self.ori_image_list.append(clean_path)
                self.matching_dict[key].append(haze_path)

        random.shuffle(self.ori_image_list)

    def __len__(self):
        return len(self.ori_image_list)

    def __getitem__(self, idx):
        ori_image_path = self.ori_image_list[idx]
        ori_filename = os.path.basename(ori_image_path)
        hazy_candidates = self.matching_dict.get(ori_filename, [])

        if not hazy_candidates:
            raise ValueError(f"No hazy images found for: {ori_filename}")

        haze_image_path = random.choice(hazy_candidates)

        print(f"\n🔍 Index {idx}")
        print(f"   - Ori image : {ori_image_path}")
        print(f"   - Haze image: {haze_image_path}")

        try:
            ori_image = Image.open(ori_image_path).convert("RGB")
            haze_image = Image.open(haze_image_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            print(f"❌ Skipping corrupted image at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transforms:
            ori_image = self.transforms(ori_image)
            haze_image = self.transforms(haze_image)

        return ori_image, haze_image


if __name__ == "__main__":
    from torchvision import transforms

    ori_dir = "./data/train/ori"
    haze_dir = "./data/train/haze"

    transform_fn = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = HazeDataset(ori_dir, haze_dir, transforms=transform_fn)

    # Load a few samples for testing
    for i in range(10):
        ori_img, haze_img = dataset[i]
