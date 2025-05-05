import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter  # ✅ Modern TensorBoard

from utils import logger, weight_init
from config import get_config
from model import AODnet
from data import HazeDataset  # ✅ Now uses random hazy variant per original


@logger
def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
        transforms.ToTensor()
    ])
    train_haze_dataset = HazeDataset(cfg.ori_data_path, cfg.haze_data_path, data_transform)
    train_loader = DataLoader(train_haze_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=False)  # ⛔ Safer on macOS

    val_haze_dataset = HazeDataset(cfg.val_ori_data_path, cfg.val_haze_data_path, data_transform)
    val_loader = DataLoader(val_haze_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                            num_workers=0, drop_last=True, pin_memory=False)

    return train_loader, len(train_loader), val_loader, len(val_loader)


@logger
def save_model(epoch, path, net, optimizer, net_name):
    model_path = os.path.join(path, net_name)
    os.makedirs(model_path, exist_ok=True)
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               os.path.join(model_path, f'AOD_{epoch}.pkl'))


@logger
def load_network(device):
    net = AODnet().to(device)
    net.apply(weight_init)
    return net


@logger
def load_optimizer(net, cfg):
    return optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


@logger
def loss_func():
    return nn.MSELoss()


@logger
def load_summaries(cfg):
    log_path = os.path.join(cfg.log_dir, cfg.net_name)
    return SummaryWriter(log_dir=log_path, comment='')


def main(cfg):
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')

    # Logging + setup
    summary = load_summaries(cfg)
    train_loader, train_number, val_loader, val_number = load_data(cfg)
    criterion = loss_func()
    network = load_network(device)
    optimizer = load_optimizer(network, cfg)

    # 🔄 Resume if possible
    start_epoch = 0
    latest_ckpt = None
    model_dir = os.path.join(cfg.model_dir, cfg.net_name)
    if os.path.exists(model_dir):
        ckpts = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if ckpts:
            latest_ckpt = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            print(f"⚡️ Resuming from checkpoint: {latest_ckpt}")
            state = torch.load(os.path.join(model_dir, latest_ckpt))
            network.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch'] + 1

    # Start training
    print('🚀 Start training')
    network.train()
    for epoch in range(start_epoch, cfg.epochs):
        for step, (ori_image, haze_image) in enumerate(train_loader):
            count = epoch * train_number + (step + 1)
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)

            dehaze_image = network(haze_image)
            loss = criterion(dehaze_image, ori_image)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            summary.add_scalar('loss', loss.item(), count)
            if step % cfg.print_gap == 0:
                summary.add_image('DeHaze_Images', make_grid(dehaze_image[:4].data, normalize=True), count)
                summary.add_image('Haze_Images', make_grid(haze_image[:4].data, normalize=True), count)
                summary.add_image('Origin_Images', make_grid(ori_image[:4].data, normalize=True), count)

            print(f"Epoch: {epoch+1}/{cfg.epochs} | Step: {step+1}/{train_number} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {loss.item():.6f}")

        # Validation image preview (10 examples max)
        print(f"Epoch: {epoch+1}/{cfg.epochs} | Saving validation outputs")
        network.eval()
        os.makedirs(cfg.sample_output_folder, exist_ok=True)
        with torch.no_grad():
            for step, (ori_image, haze_image) in enumerate(val_loader):
                if step > 10:
                    break
                ori_image, haze_image = ori_image.to(device), haze_image.to(device)
                dehaze_image = network(haze_image)
                grid = make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0), nrow=ori_image.size(0))
                save_path = os.path.join(cfg.sample_output_folder, f'{epoch+1}_{step}.jpg')
                save_image(grid, save_path)
        network.train()

        # Save model
        save_model(epoch, cfg.model_dir, network, optimizer, cfg.net_name)

    summary.close()
    print("✅ Training complete.")


if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
