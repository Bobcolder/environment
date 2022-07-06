import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import yaml
import tqdm
import logging

import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)  # 必要时检查，有时要进入脚本所在目录运行

from utils import set_seed, value2class, inverse_PM, dataset_class_count
from datasets import get_transform
from networks import get_nets
from metric_counter import MetricCounter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(1)  # set different random seed


class TrainerMul:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.metric_counter = MetricCounter(config['experiment_desc'])
        self.pretrained = config['train']['continue_train']
        self.start_epoch = 0

    def train(self):
        self._init_params()
        if self.pretrained:
            self.start_epoch = self.config['train']['start_epoch']
            self.model.load_state_dict(torch.load(self.config['train']['model_path'], map_location=device)['model'])

        self.model.to(device)
        for epoch in range(self.start_epoch, self.start_epoch + self.config['num_epochs']):
            self._run_epoch(epoch)
            self._run_epoch(epoch, valid=True)

            if self.metric_counter.update_best_model(self.best_metric):
                torch.save({
                    'model': self.model.state_dict()
                }, '{}/best_{}.h5'.format(self.save_dir, self.config['experiment_desc']))
                self.metric_counter.write_CMatrix(os.path.join(self.save_dir, self.config['experiment_desc']))
            torch.save({
                'model': self.model.state_dict()
            }, '{}/last_{}.h5'.format(self.save_dir, self.config['experiment_desc']))

            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                '{}/{}'.format(self.save_dir, self.config['experiment_desc']),
                epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch, valid=False):
        self.metric_counter.clear()
        self.model.train(not valid)  # 区分训练还是验证阶段
        run_dataset = self.train_dataset if not valid else self.val_dataset

        epoch_size = min(self.config.get('batches_per_epoch'), len(run_dataset))
        tq = tqdm.tqdm(run_dataset, total=epoch_size)
        tq.set_description('Epoch {}'.format(epoch) if not valid else "Validation")
        i = 0
        for data in tq:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if not valid:
                self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            if not valid:
                loss.backward()
                self.optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            correct = torch.sum(preds == labels).item()
            acc = correct * 1. / labels.size(0)

            self.metric_counter.add_losses(('CrossEntropyLoss',), (loss.item(),))
            self.metric_counter.add_metrics(('Acc',), (acc,))
            self.metric_counter.add_CMatrix(labels.cpu().data.numpy(), preds.cpu().data.numpy())
            tq.set_postfix(loss_acc=self.metric_counter.loss_message())  # metric_counter 内求平均值
           
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=valid)

    def _get_optim(self, params):
        lr = self.config['optimizer']['lr']
        momentum = self.config['optimizer']['momentum']
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=lr, momentum=momentum)
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=lr, momentum=momentum)
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=lr, momentum=momentum)
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _init_params(self):
        self.criterion = nn.CrossEntropyLoss()  # get_loss 抽象
        self.model = get_nets(self.config['model']['g_name'], self.config['model']['out_features'])
        self.optimizer = self._get_optim(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.best_metric = 'Acc'
        self.save_dir = self.config['experiment_desc']
        if not os.path.exists(os.path.join(curPath, self.save_dir)):
            os.makedirs(os.path.join(curPath, self.save_dir))
    

if __name__ == "__main__":
    with open('networks/config/config.yaml', 'r') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        data_dir = config_list['nonsky_dir']
        batch_size = config_list['batch_size']
        imgsize = config_list['image_size']
        train_trans_first = config_list['train']['transform_first']

    train_dir = os.path.join(data_dir, 'train_blocks')
    train_transform = get_transform(imgsize, train_trans_first)
    full_dataset = ImageFolder(train_dir, transform=train_transform)
    dataset_class_count(full_dataset)

    # Split the indices in a stratified way
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    trainer = TrainerMul(config_list, train=train_loader, val=val_loader)
    trainer.train()








