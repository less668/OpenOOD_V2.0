import csv
import os
from typing import Dict, List

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from openood.postprocessors import BasePostprocessor
from openood.utils import Config
import pickle
from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (15,10)
import copy
import torch
class OODEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(OODEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None

    def eval_ood(self,
                 net: nn.Module,
                 id_data_loaders: Dict[str, DataLoader],
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor,
                 fsood: bool = False):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name

        if self.config.postprocessor.APS_mode:
            assert 'val' in id_data_loaders
            assert 'val' in ood_data_loaders
            self.hyperparam_search(net, id_data_loaders['val'],
                                   ood_data_loaders['val'], postprocessor)
        # sys.exit()
        print(f'Performing inference on {dataset_name} dataset...', flush=True)

        try:
            if postprocessor.postprocessorname == 'SPCA':
                id_pred, id_conf, id_gt, _ = postprocessor.inference(
                    net, id_data_loaders['test'],pca_obj=self.id_pca, pca_fit=0)  #test
        except:
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loaders['test'])

        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        if fsood:
            # load csid data and compute confidence
            for dataset_name, csid_dl in ood_data_loaders['csid'].items():
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                if self.config.recorder.save_scores:
                    self._save_scores(csid_pred, csid_conf, csid_gt,
                                      dataset_name)
                id_pred = np.concatenate([id_pred, csid_pred])
                id_conf = np.concatenate([id_conf, csid_conf])
                id_gt = np.concatenate([id_gt, csid_gt])

        # load nearood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')

        # load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')

    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood'):
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)

            try:
                if postprocessor.postprocessorname == 'SPCA':
                    #print("This is inside _eval_ood : PCA singular values :", self.id_pca.singular_values_,flush=True)
                    ood_pred, ood_conf, ood_gt, _ = postprocessor.inference(net, ood_dl,pca_obj=self.id_pca, pca_fit=0)

                    number_classes = int(np.max(id_pred) + 1)
                    hard_classes = []
                    for i in range(number_classes):
                        id_ids = np.where(id_pred == i)[0]
                        ood_ids = np.where(ood_pred == i)[0]

                        # print(id_ids.shape,ood_ids.shape)
                        # print(id_gt.shape,ood_gt.shape)

                        pred_id = id_pred[id_ids]
                        pred_ood = ood_pred[ood_ids]
                        conf_id = id_conf[id_ids]
                        conf_ood = ood_conf[ood_ids]
                        gt_id = id_gt[id_ids]
                        gt_ood = ood_gt[ood_ids]

                        gt_ood = -1 * np.ones_like(gt_ood)  # hard set to -1 as ood

                        pred = np.concatenate([pred_id, pred_ood])
                        conf = np.concatenate([conf_id, conf_ood])
                        label = np.concatenate([gt_id, gt_ood])

                        # print("pred",pred)
                        # print("conf", conf)
                        # print("label",label)

                        ood_metrics = compute_all_metrics(conf, label, pred)
                        fpr = ood_metrics[0]
                        if fpr > 0.5:
                            hard_classes.append(i)

                        plt.figure()
                        plt.hist(conf_id, label="ID",alpha=0.3)
                        plt.hist(conf_ood, color='g', label="OOD", alpha=0.3)
                        plt.legend(fontsize=15)
                        plt.tick_params(labelsize=15)
                        plt.xlabel("JSI Score", fontsize=15)
                        plt.ylabel("Frequency", fontsize=15)
                        plt.title("Class "+str(i)+" Pred Strength Score for ID and OOD ("+dataset_name+") data "+"FPR : "+str(np.round(fpr,3)) , fontsize=15)
                        # plt.savefig("./plots5/"+dataset_name+"/Class "+str(i)+" ID and OOD ("+dataset_name+").png")
                        plt.close()
                    print(dataset_name+" Hard classes - "+str(len(hard_classes)),hard_classes)

                    res = 1e-2
                    g_min = min(np.min(id_conf), np.min(ood_conf))
                    g_max = max(np.max(id_conf), np.max(ood_conf))

                    g_range = (g_min, g_max)
                    n_bins = int((g_max - g_min) / res)
                    plt.figure()
                    plt.hist(id_conf, label="ID", alpha=0.3, bins=n_bins, range=g_range)
                    plt.hist(ood_conf, color='g', label="OOD", alpha=0.3, bins=n_bins, range=g_range)
                    plt.legend(fontsize=15)
                    plt.tick_params(labelsize=15)
                    plt.xlabel("Prediction Strength Score", fontsize=15)
                    plt.ylabel("Frequency", fontsize=15)
                    plt.title("Prediction Strength Score Distribution for ID and OOD ("+dataset_name+") data", fontsize=15)
                    # plt.savefig("ID and OOD ("+dataset_name+").png")
                    plt.close()

            except:
                ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            # print(dataset_name+" Analysis: Max/Min/Mean")
            # print("ID :",np.max(id_conf), np.min(id_conf), np.mean(id_conf))
            # print("OOD :",np.max(ood_conf), np.min(ood_conf), np.mean(ood_conf))


            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)

    def eval_ood_val(self, net: nn.Module, id_data_loaders: Dict[str,
                                                                 DataLoader],
                     ood_data_loaders: Dict[str, DataLoader],
                     postprocessor: BasePostprocessor):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'val' in id_data_loaders
        assert 'val' in ood_data_loaders
        if self.config.postprocessor.APS_mode:
            val_auroc = self.hyperparam_search(net, id_data_loaders['val'],
                                               ood_data_loaders['val'],
                                               postprocessor)
        else:
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loaders['val'])
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loaders['val'])
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            val_auroc = ood_metrics[1]
        return {'auroc': 100 * val_auroc}

    def _save_csv(self, metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'ACC': '{:.2f}'.format(100 * accuracy)
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 fsood: bool = False,
                 csid_data_loaders: DataLoader = None):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()

        try:
            if postprocessor.postprocessorname == 'SPCA':
                print("Postprocessor Name : ",postprocessor.postprocessorname,flush=True)   
                self.id_pred, self.id_conf, self.id_gt, _ = postprocessor.inference(
                    net, data_loader,pca_obj=None, pca_fit=0)  #previously, self.id_pca_orig last variable and pca_fit=1
                with open("cifar10_CLM_o.pickle", 'rb') as file:
                    # Load data from the pickle file
                    pred_pdf = pickle.load(file)
                self.id_pca_orig = pred_pdf
                # self.id_pred, self.id_conf, self.id_gt, _ = postprocessor.inference(
                #     net, data_loader, pca_obj=self.id_pca, pca_fit=False)

        except:
            self.id_pred, self.id_conf, self.id_gt = postprocessor.inference(
                net, data_loader)




        if fsood:
            assert csid_data_loaders is not None
            for dataset_name, csid_dl in csid_data_loaders.items():
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                self.id_pred = np.concatenate([self.id_pred, csid_pred])
                self.id_conf = np.concatenate([self.id_conf, csid_conf])
                self.id_gt = np.concatenate([self.id_gt, csid_gt])

        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_data_loader,
        ood_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print('Starting automatic parameter search...')
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0
        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)

            try:
                if postprocessor.postprocessorname == 'SPCA':
                    id_pca_copy = copy.deepcopy(self.id_pca_orig)
                    t = 0.03
                    top_k = 10
                    target_matrix = np.zeros((10, 10, top_k))
                    for i in id_pca_copy:
                        for j in range(top_k):
                            for k in id_pca_copy[i][j]:
                                if id_pca_copy[i][j][k] >= t:
                                    target_matrix[i][k][j] = id_pca_copy[i][j][k]

                    column_sums = target_matrix.sum(axis=1, keepdims=True)

                    zero_sum_columns = (column_sums == 0)
                    column_sums[zero_sum_columns] = 1  # To avoid division by zero
                    scaled_matrix = target_matrix / column_sums

                    for i in range(target_matrix.shape[0]):
                        for k in range(target_matrix.shape[2]):
                            if zero_sum_columns[i, 0, k]:
                                scaled_matrix[i, :, k] = 1 / top_k
                    self.id_pca = torch.from_numpy(scaled_matrix).to('cuda:0')
                    print("Object saved")

                    # print("original", id_pca_copy[0][2])
                    # _, _, _, pca_object = postprocessor.inference(
                    #     net, id_data_loader, pca_obj=id_pca_copy, pca_fit=2)
                    # print(hyperparam, pca_object[0][2],flush=True)
                    id_pred, id_conf, id_gt, _ = postprocessor.inference(
                        net, id_data_loader, pca_obj=self.id_pca, pca_fit=0)
                    ood_pred, ood_conf, ood_gt, _ = postprocessor.inference(
                        net, ood_data_loader, pca_obj=self.id_pca, pca_fit=0)
            except:
                id_pred, id_conf, id_gt = postprocessor.inference(
                    net, id_data_loader)
                ood_pred, ood_conf, ood_gt = postprocessor.inference(
                    net, ood_data_loader)

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print('Hyperparam:{}, auroc:{}'.format(hyperparam,
                                                   aps_dict[index]))
            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
                try:
                    if postprocessor.postprocessorname == 'SPCA':
                        pass
                        # t = 0.05
                        # top_k = 100
                        # target_matrix = np.zeros((100, 100, top_k))
                        # for i in id_pca_copy:
                        #     for j in range(top_k):
                        #         for k in id_pca_copy[i][j]:
                        #             if id_pca_copy[i][j][k] >= t:
                        #                 target_matrix[i][k][j] = 1  # data[i][j][k]
                        #
                        # column_sums = target_matrix.sum(axis=1, keepdims=True)
                        #
                        # zero_sum_columns = (column_sums == 0)
                        # column_sums[zero_sum_columns] = 1  # To avoid division by zero
                        # scaled_matrix = target_matrix / column_sums
                        #
                        # for i in range(target_matrix.shape[0]):
                        #     for k in range(target_matrix.shape[2]):
                        #         if zero_sum_columns[i, 0, k]:
                        #             scaled_matrix[i, :, k] = 1 / top_k
                        # self.id_pca = torch.from_numpy(scaled_matrix).to('cuda:0')
                        # print("Object saved")
                        # self.id_pca = pca_object

                except:
                    pass
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print('Final hyperparam: {}'.format(postprocessor.get_hyperparam()))
        return max_auroc

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
        return results


CraftOODEvaluator = OODEvaluator
