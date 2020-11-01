import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data.segmentation import SegmentDataset
from model.segmentation.fcn import FCN32
from model.segmentation.unet import UNet, UNetVGG16


__all__ = [ "SegmentAgent" ]

class SegmentAgent:
    """Train Image Segmentation model

    Requirements:
        Simple baseline
            - (15%) validation mIoU > 0.635
            - (15%) testing mIoU > 0.625
    """
    def __init__(self, config):
        self.config = config

        # Check environment
        if torch.cuda.is_available():
            self.device = torch.device(config['train']['device'])
        else:
            raise RuntimeError("Please train your model with GPU")

        # Create dataset
        tr_transform = T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]), ])
        te_transform = T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]), ])
        train_dataset = SegmentDataset(root=config['dataset']['train']['root'],
                                        transform=tr_transform)
        valid_dataset = SegmentDataset(root=config['dataset']['valid']['root'],
                                        transform=te_transform)

        # Create dataloader
        self.train_loader = DataLoader(train_dataset,
                                batch_size=config['loader']['batch_size'],
                                num_workers=config['loader']['num_workers'],
                                shuffle=True)
        self.valid_loader = DataLoader(valid_dataset,
                                batch_size=config['loader']['batch_size'],
                                num_workers=config['loader']['num_workers'],
                                shuffle=False)

        # Create model
        if config['train']['model'] == 'fcn':
            self.model = FCN32(n_classes=7)
        elif config['train']['model'] == 'unet':
            self.model = UNetVGG16(n_classes=7)
        self.model.to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['optim']['lr'])

        # Create loss function
        self.criterion = nn.CrossEntropyLoss()

        # Create tensorboard
        tensorboard_dir = osp.join(config['train']['log_dir'], config['train']['exp_name'])
        self.writer = SummaryWriter(tensorboard_dir)

        # Logging
        self.start_epoch = 0
        self.current_epoch = -1
        self.current_loss = 10000

        # Resume training or not
        if config['train']['resume']:
            checkpoint_file = osp.join(config['train']['log_dir'],
                                    config['train']['checkpoint_dir'],
                                    'best.pth')
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = config['optim']['lr']
            self.current_epoch = checkpoint['current_epoch'] + 1
            self.start_epoch = self.current_epoch + 1
            print("Resume training at epoch {}".format(self.start_epoch))

    def train(self):
        for epoch in range(self.start_epoch, self.config['train']['n_epochs']):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()

    def train_one_epoch(self):
        running_loss = 0

        self.model.train()
        for i, (imgs, targets) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            # Forward & Backward
            self.optimizer.zero_grad()
            outputs = self.model(imgs) # (n, c, h, w)
            preds = outputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 7)
            labels = targets.flatten()
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

            # Cumulate result
            running_loss += loss.item() * len(imgs)

            # Show training information
            if (i % self.config['train']['interval']) == 0:
                print("Epoch {}:{}({}%), Loss: {:.2f}".format(
                    self.current_epoch, self.config['train']['n_epochs'],
                    int(i*100/len(self.train_loader)), loss.item()))

        train_loss = running_loss / len(self.train_loader.dataset)
        print("Epoch {}:{}, Train Loss: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'], train_loss))

        # Export result to tensorboard
        self.writer.add_scalar("Train Loss", train_loss, self.current_epoch)

    def validate(self):
        running_loss = 0
        pred_masks = []
        true_masks = []

        self.model.eval()
        with torch.no_grad():
            for imgs, targets in self.valid_loader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(imgs) # (n, c, h, w)

                # Save segmenation mask
                pred_mask = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                pred_masks.append(pred_mask)
                true_masks.append(targets.detach().cpu().numpy())

                # Compute loss
                preds = outputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 7)
                labels = targets.flatten()
                loss = self.criterion(preds, labels)

                # Validation Loss
                running_loss += loss.item() * len(imgs)

        # Show validation result
        pred_masks = np.vstack(pred_masks)
        true_masks = np.vstack(true_masks)
        miou = self._mean_iou_score(pred_masks, true_masks)
        valid_loss = running_loss / len(self.valid_loader.dataset)
        print("Epoch {}:{}, Valid Loss: {:.2f}, mIoU: {:.3f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            valid_loss, miou))

        # Save training checkpoints
        if valid_loss < self.current_loss:
            self.current_loss = valid_loss
            self._save_checkpoint()

        # Export result to tensorboard
        self.writer.add_scalar("Valid Loss", valid_loss, self.current_epoch)

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoints = { 'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'current_epoch': self.current_epoch,
                        'current_loss': self.current_loss }
        checkpoint_file = osp.join(self.config['train']['log_dir'],
                                self.config['train']['checkpoint_dir'],
                                'best.pth')
        if not osp.exists(osp.dirname(checkpoint_file)):
            os.makedirs(osp.dirname(checkpoint_file))

        torch.save(checkpoints, checkpoint_file)
        print("Save checkpoint to '{}'".format(checkpoint_file))

    def _mean_iou_score(self, pred_masks, true_masks):
        """Compute mean IoU score over 6 classes"""
        mean_iou = 0
        for i in range(6):
            tp_fp = np.sum(pred_masks == i)
            tp_fn = np.sum(true_masks == i)
            tp = np.sum((pred_masks == i) * (true_masks == i))
            iou = tp / (tp_fp + tp_fn - tp)
            mean_iou += iou / 6
        return mean_iou
