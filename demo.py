#!/usr/bin/env python3
import os
import sys
import glob
import torch
from torchvision import transforms, utils
from PIL import Image
from utils import logger
from config import get_config
from model import AODnet

@logger
def make_test_data(cfg, img_path_list, device):
    from torchvision import transforms
    data_transform = transforms.Compose([
        transforms.ToTensor()  
    ])
    imgs = []
    for img_path in img_path_list:
        try:
            image = Image.open(img_path).convert("RGB")
            x = data_transform(image).unsqueeze(0).to(device)
            imgs.append((x, img_path))
        except Exception as e:
            print(f"[⚠️] Failed to load image: {img_path} — {e}")
    return imgs


@logger
def load_pretrain_network(cfg, device):
    model_path = os.path.join(cfg.model_dir, cfg.ckpt)
    print(f"🔄 Loading model from: {model_path}")
    net = AODnet().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    net.load_state_dict(state_dict)
    return net

def main(cfg):
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')

    # gather files
    pattern = os.path.join(cfg.test_img_dir, '*')
    test_paths = [p for p in glob.glob(pattern) if p.lower().endswith(('.jpg','.jpeg','.png'))]
    test_data  = make_test_data(cfg, test_paths, device)

    # load and eval
    net = load_pretrain_network(cfg, device)
    net.eval()
    os.makedirs(cfg.sample_output_folder, exist_ok=True)
    print('🚀 Running inference...')

    for tensor, path in test_data:
        with torch.no_grad():
            out = net(tensor)
        fname = os.path.basename(path)
        save_path = os.path.join(cfg.sample_output_folder, fname)
        # only save the dehazed output, one image per file
        utils.save_image(out, save_path, nrow=1)
        print(f"✅ Saved dehazed: {save_path}")

    print("🎉 All done!")

if __name__ == '__main__':
    cfg, _ = get_config()
    # override via CLI: --test_img_dir /my/folder
    for i, a in enumerate(sys.argv):
        if a == '--test_img_dir' and i+1 < len(sys.argv):
            cfg.test_img_dir = sys.argv[i+1]
    main(cfg)



    config_args.test_img_dir = args.test_img_dir
    main(config_args)

