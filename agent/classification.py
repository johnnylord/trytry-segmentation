import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data.classification import ImageDataset
from model.classification.vgg16 import VGG16


__all__ = [ "ClassAgent" ]

class ClassAgent:
    """Train image classification model

    Requirements:
        - (15%) validation accuracy should > 0.7
        - (5%) testing accuracy should > 0.7
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
                            T.RandomHorizontalFlip(),
                            T.RandomRotation(10),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]), ])
        te_transform = T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]), ])
        train_dataset = ImageDataset(root=config['dataset']['train']['root'],
                                    transform=tr_transform)
        valid_dataset = ImageDataset(config['dataset']['valid']['root'],
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
        self.model = VGG16(n_classes=50)
        self.model.to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['optim']['lr'])

        # Create loss function
        self.criterion = nn.CrossEntropyLoss()

        # Create tensorboard
        tensorboard_dir = osp.join(config['train']['log_dir'],
                                config['train']['exp_name'])
        self.writer = SummaryWriter(tensorboard_dir)

        # Logging
        self.start_epoch = 0
        self.current_epoch = -1
        self.current_loss = 10000

        # Resume training
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
        running_corrects = 0

        self.model.train()
        for i, (imgs, labels) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            # Forward & Backward
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Accuracy
            preds = torch.max(outputs.data, 1)[1]
            corrects = float(torch.sum(preds == labels.data))
            acc = corrects / len(preds)

            # Cumulate result
            running_loss += loss.item() * len(imgs)
            running_corrects += corrects

            # Show training information
            if (i % self.config['train']['interval']) == 0:
                print("Epoch {}:{}({}%), Loss: {:.2f}, Acc: {:.2f}".format(
                    self.current_epoch, self.config['train']['n_epochs'],
                    int(i*100/len(self.train_loader)), loss.item(), acc))

        train_loss = running_loss / len(self.train_loader.dataset)
        train_acc = running_corrects / len(self.train_loader.dataset)
        print("Epoch {}:{}, Train Loss: {:.2f}, Train Acc: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            train_loss, train_acc))

        # Export result to tensorboard
        self.writer.add_scalar("Train Loss", train_loss, self.current_epoch)
        self.writer.add_scalar("Train Acc", train_acc, self.current_epoch)

    def validate(self):
        running_loss = 0
        running_corrects = 0
        self.model.eval()
        for imgs, labels in self.valid_loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            preds = torch.max(outputs.data, 1)[1]
            corrects = float(torch.sum(preds == labels.data))

            running_loss += loss.item() * len(imgs)
            running_corrects += corrects

        valid_loss = running_loss / len(self.valid_loader.dataset)
        valid_acc = running_corrects / len(self.valid_loader.dataset)
        print("Epoch {}:{}, Valid Loss: {:.2f}, Valid Acc: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            valid_loss, valid_acc))

        # Save training checkpoints
        if valid_loss < self.current_loss:
            self.current_loss = valid_loss
            self._save_checkpoint()

        # Export result to tensorboard
        self.writer.add_scalar("Valid Loss", valid_loss, self.current_epoch)
        self.writer.add_scalar("Valid Acc", valid_acc, self.current_epoch)

    def finalize(self):
        print("Finish training process")

    def _save_checkpoint(self):
        checkpoints = { 'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'current_epoch': self.current_epoch }

        checkpoint_file = osp.join(self.config['train']['log_dir'],
                                self.config['train']['checkpoint_dir'],
                                'best.pth')
        if not osp.exists(osp.dirname(checkpoint_file)):
            os.makedirs(osp.dirname(checkpoint_file))

        torch.save(checkpoints, checkpoint_file)
        print("Save checkpoint to '{}'".format(checkpoint_file))
