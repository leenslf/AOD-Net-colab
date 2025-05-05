# import os
# import glob
# import torch
# from torchvision import transforms, utils
# from PIL import Image
# from utils import logger
# from config import get_config
# from model import AODnet


# @logger
# def make_test_data(cfg, img_path_list, device):
#     data_transform = transforms.Compose([
#         transforms.Resize([480, 640]),
#         transforms.ToTensor()
#     ])
#     imgs = []
#     for img_path in img_path_list:
#         try:
#             image = Image.open(img_path).convert("RGB")
#             x = data_transform(image).unsqueeze(0).to(device)
#             imgs.append(x)
#         except Exception as e:
#             print(f"[⚠️] Failed to load image: {img_path} — {e}")
#     return imgs


# @logger
# def load_pretrain_network(cfg, device):
#     model_path = os.path.join(cfg.model_dir, cfg.net_name, cfg.ckpt)
#     print(f"🔄 Loading model from: {model_path}")
    
#     net = AODnet().to(device)
#     ckpt = torch.load(model_path, map_location=device)
    
#     # Compatibility: if it's just the state_dict or a full checkpoint
#     state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
#     net.load_state_dict(state_dict)
    
#     return net


# def main(cfg):
#     print(cfg)
#     if cfg.gpu > -1:
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
#     device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')

#     # -------------------------------------------------------------------
#     # Load test images
#     test_file_path = glob.glob('./test_images/*.jpg')
#     test_images = make_test_data(cfg, test_file_path, device)

#     # -------------------------------------------------------------------
#     # Load network
#     network = load_pretrain_network(cfg, device)

#     # -------------------------------------------------------------------
#     # Run inference
#     print('🚀 Starting evaluation...')
#     network.eval()
#     os.makedirs("results", exist_ok=True)

#     for idx, im in enumerate(test_images):
#         with torch.no_grad():
#             dehaze_image = network(im)
#         save_path = os.path.join("results", os.path.basename(test_file_path[idx]))
#         utils.save_image(torch.cat((im, dehaze_image), 0), save_path)
#         print(f"✅ Saved: {save_path}")

#     print("🎉 Done.")


# if __name__ == '__main__':
#     config_args, unparsed_args = get_config()
#     main(config_args)

import os
import glob
import torch
from torchvision import transforms, utils
from PIL import Image
from utils import logger
from config import get_config
from model import AODnet

@logger
def make_test_data(cfg, img_path_list, device):
    data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
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

    # Compatibility: support either raw state_dict or full checkpoint
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    net.load_state_dict(state_dict)
    return net

def main(cfg):
    print(cfg)

    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')

    # Load test images (supports .jpg, .jpeg, .png)
    test_file_path = glob.glob(os.path.join(cfg.test_img_dir, '*.jpg')) + \
                     glob.glob(os.path.join(cfg.test_img_dir, '*.jpeg')) + \
                     glob.glob(os.path.join(cfg.test_img_dir, '*.png'))

    test_images = make_test_data(cfg, test_file_path, device)

    # Load model
    network = load_pretrain_network(cfg, device)

    # Inference
    print('🚀 Starting evaluation...')
    os.makedirs(cfg.sample_output_folder, exist_ok=True)
    network.eval()

    for idx, (im, path) in enumerate(test_images):
        with torch.no_grad():
            dehaze_image = network(im)

        filename = os.path.basename(path)
        save_path = os.path.join(cfg.sample_output_folder, filename)
        utils.save_image(torch.cat((im, dehaze_image), 0), save_path)
        print(f"✅ Saved: {save_path}")

    print("🎉 Done.")

if __name__ == '__main__':
    config_args, _ = get_config()

    # Inject custom test path if provided via CLI
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == '--test_img_dir' and i + 1 < len(sys.argv):
            config_args.test_img_dir = sys.argv[i + 1]

    main(config_args)


    config_args.test_img_dir = args.test_img_dir
    main(config_args)

