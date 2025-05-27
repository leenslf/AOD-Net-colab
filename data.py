import os
import random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class HazeDataset(Dataset):
    def __init__(self, ori_root, haze_root, transforms=None, verbose=False):
        """
        Dataset for paired clean/hazy images with matching filenames.
        :param ori_root: Path to ground truth (clean) images
        :param haze_root: Path to hazy images
        :param transforms: Optional image transforms
        :param verbose: Whether to print per-sample info (default: False)
        """
        self.ori_root = ori_root
        self.haze_root = haze_root
        self.transforms = transforms
        self.verbose = verbose

        self.image_filenames = sorted([
            f for f in os.listdir(haze_root)
            if f.lower().endswith(('.jpg', '.png')) and os.path.exists(os.path.join(ori_root, f))
        ])

        if not self.image_filenames:
            raise RuntimeError(f"❌ No matching image pairs found in {haze_root} and {ori_root}")

        print(f"✅ Matched image pairs: {len(self.image_filenames)}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        ori_path = os.path.join(self.ori_root, filename)
        haze_path = os.path.join(self.haze_root, filename)

        try:
            ori_image = Image.open(ori_path).convert("RGB")
            haze_image = Image.open(haze_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            print(f"❌ Skipping corrupted image: {filename} — {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.verbose:
            print(f"\n🔍 Index {idx}")
            print(f"   - Ori image : {ori_path}")
            print(f"   - Haze image: {haze_path}")

        if self.transforms:
            ori_image = self.transforms(ori_image)
            haze_image = self.transforms(haze_image)

        return ori_image, haze_image


if __name__ == "__main__":
    from torchvision import transforms

    ori_dir = "./data/train/reference"
    haze_dir = "./data/train/raw"

    transform_fn = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = HazeDataset(ori_dir, haze_dir, transforms=transform_fn, verbose=True)

    for i in range(10):
        ori_img, haze_img = dataset[i]
        print(f"[{i}] Shapes — Ori: {ori_img.shape}, Haze: {haze_img.shape}")
