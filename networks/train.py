import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import yaml
import tqdm
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error

from utils import set_seed, value2class, inverse_PM
from datasets import ImagePMSet, get_transform
from networks import get_nets
from metric_counter import MetricCounter

curPath = os.path.abspath(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(1)  # set different random seed


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.metric_counter = MetricCounter(config['experiment_desc'])

    def train(self):
        self._init_params()
        self.model.to(device)
        for epoch in range(0, self.config['num_epochs']):
            self._run_epoch(epoch)
            self._run_epoch(epoch, valid=True)

            if self.metric_counter.update_best_model(self.best_metric):
                torch.save({
                    'model': self.model.state_dict()
                }, '{}/best_{}.h5'.format(self.save_dir, self.config['experiment_desc']))
            torch.save({
                'model': self.model.state_dict()
            }, '{}/last_{}.h5'.format(self.save_dir, self.config['experiment_desc']))

            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                '{}/{}'.format(self.save_dir, self.config['experiment_desc']),
                epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch, valid=False):
        self.metric_counter.clear()
        self.model.train(not valid)
        run_dataset = self.train_dataset if not valid else self.val_dataset

        epoch_size = min(self.config.get('batches_per_epoch'), len(run_dataset))
        tq = tqdm.tqdm(run_dataset, total=epoch_size)
        tq.set_description('Epoch {}'.format(epoch) if not valid else "Validation")
        i = 0
        for data in tq:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            
            if not valid:
                self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            if not valid:
                loss.backward()
                self.optimizer.step()

            outputs, labels = outputs.cpu().data.numpy(), labels.cpu().data.numpy()
            outputs, labels = map(inverse_PM, (outputs, labels))  # bug
            mae = mean_absolute_error(labels, outputs)
            outputs, labels = map(value2class, (outputs, labels))
            correct = np.sum(outputs == labels)
            acc = correct * 1. / labels.shape[0]

            self.metric_counter.add_losses(('MSELoss',), (loss.item(),))
            self.metric_counter.add_metrics(('Acc', 'MAE'), (acc, mae))
            # metric_counter 内求平均值
            tq.set_postfix(loss_acc=self.metric_counter.loss_message())
           
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
        self.criterion = nn.MSELoss()  # get_loss 抽象
        self.model = get_nets(self.config['model'])
        self.optimizer = self._get_optim(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.best_metric = 'Acc'
        self.save_dir = self.config['experiment_desc']
        if not os.path.exists(os.path.join(curPath, self.save_dir)):
            os.makedirs(os.path.join(curPath, self.save_dir))
    

if __name__ == "__main__":
    with open('./config/config.yaml', 'r') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        data_dir = config_list['nonsky_dir']
        batch_size = config_list['batch_size']
        imgsize = config_list['image_size']
        train_trans_first = config_list['train']['transform_first']
        val_trans_first = config_list['val']['transform_first']

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "val")
    train_transform = get_transform(imgsize, train_trans_first)
    valid_transform = get_transform(imgsize, val_trans_first)
    train_dataset = ImagePMSet(train_dir, transform=train_transform)
    valid_dataset = ImagePMSet(valid_dir, transform=valid_transform)
    # 先记录下来，后面再写入文件. 45.6318, 18.3280
    PM_mean, PM_std = train_dataset.get_mean_std()

    # 后续若 Subset 使用不同的 transform 如何处理？
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    trainer = Trainer(config_list, train=train_loader, val=valid_loader)
    trainer.train()








