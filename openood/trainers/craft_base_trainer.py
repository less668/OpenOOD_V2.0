import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import sys
import openood.utils.comm as comm
from openood.utils import Config
from matplotlib import pyplot as plt

from .lr_scheduler import cosine_annealing
with open("cifar10_CLM_o.pickle", 'rb') as file:
    # Load data from the pickle file
    data = pickle.load(file)
t = 0.03
top_k = 10
target_matrix = np.zeros((10,10,top_k))
for i in data:
    for j in range(top_k):
        for k in data[i][j]:
            if data[i][j][k] >= t:
                target_matrix[i][k][j] = data[i][j][k]

column_sums = target_matrix.sum(axis=1,keepdims=True)

zero_sum_columns = (column_sums == 0)
column_sums[zero_sum_columns] = 1  # To avoid division by zero
scaled_matrix = target_matrix / column_sums

for i in range(target_matrix.shape[0]):
    for k in range(target_matrix.shape[2]):
        if zero_sum_columns[i, 0, k]:
            scaled_matrix[i, :, k] = 1/top_k

# output_dir = 'cifar100_original_plots_t10_100'
# scaled_2 = scaled_matrix
# for i in range(100):
#     print(target_matrix[i])
#     scaled_2[scaled_2 == 0] = np.nan
#     plt.imshow(scaled_2[i], cmap='viridis', interpolation='none')
#     plt.colorbar()
#     plt.title(f'Class {i}')
#     plt.savefig(f'{output_dir}/matrix_{i}.png')
#     plt.close()  # Close the figure to free memory
#     print(np.round(scaled_matrix[i],2))
#     print(np.sum(scaled_matrix,axis=1))
# sys.exit()
rank_ground_truth = torch.from_numpy(scaled_matrix).to('cuda:0')


class BaseTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        primary_loss_avg = 0.0
        secondary_loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()
            # print(target)
            # print(target.shape)
            rank_pred = rank_ground_truth[target].float()
            # print("rank_pred shape", rank_pred)
            # forward
            logits_classifier = self.net(data)
            ### New addition for class-aware training
            num_classes = int(logits_classifier.shape[1]/top_k)
            logits_classifier_new = logits_classifier.view(logits_classifier.shape[0],num_classes,top_k)
            # print(logits_class_new.shape) (batch_size,10,3) for cifar10 top 3
            # sys.exit()

            primary_loss = F.cross_entropy(logits_classifier_new[:,:,0], torch.max(rank_pred[:,:,0],dim=1)[1]) #loss = F.cross_entropy(logits_classifier, target)
            # print(primary_loss,type(primary_loss)) # 1.xxxx
            
            secondary_loss = 0.0
            for kk in range(1,top_k):
                logits_kk = logits_classifier_new[:,:,kk]
                log_prob_logits_kk = F.log_softmax(logits_kk,dim=1)
                sec_kl_div = F.kl_div(log_prob_logits_kk, rank_pred[:,:,kk])
                secondary_loss += sec_kl_div

            secondary_loss = secondary_loss.float()  #0.xxxx
            # alpha = 0.8
            # print(secondary_loss,type(secondary_loss))
            total_loss =   primary_loss + 20*secondary_loss
            # print(total_loss,type(total_loss))

            # backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(total_loss) * 0.2
                primary_loss_avg = primary_loss_avg * 0.8 + float(primary_loss) * 0.2
                secondary_loss_avg = secondary_loss_avg * 0.8 + float(secondary_loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)
        metrics['primary_loss'] = self.save_metrics(primary_loss_avg)
        metrics['secondary_loss'] = self.save_metrics(secondary_loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
