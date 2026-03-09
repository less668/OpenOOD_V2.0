from typing import Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
import openood.utils.comm as comm
from .base_postprocessor import BasePostprocessor
from sklearn.decomposition import PCA
import numpy as np
from torchsummary import summary
import sys
import scipy

class SPCAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        # self.pcacomponents = self.args.pcacomponents
        self.a = self.args.a
        self.upper = self.args.upper
        self.reward = self.args.reward
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.postprocessorname = "SPCA"
        self.topk = 10

    @torch.no_grad()
    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  pca_obj: None,
                  pca_fit: int,
                  progress: bool = True):
        pred_list, conf_list, label_list, logits_list = [], [], [], []
        for batch in tqdm(data_loader,disable=False):
                          #disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf, logits = self.postprocess(net, data, pca_obj)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())
            logits_list.append(logits.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        logits_list = torch.cat(logits_list).numpy()



        if pca_fit == 1:
            # print(logits_list.shape)
            # np.save("imagenet_logits", logits_list)
            ####### pca_fit == 1 computes the original probabilities of the class likelihood matrics.
            ####### pca_fit == 2 smoothes the class likelihood matrices. This is used for hyperparamter tuning.

            ###### Logits list is a numpy tensor with shape (No. of samples, No.of classes) #######
            # logits_list = np.load("imagenet_logits.npy")
            # print("Numpy file loading completed...", flush=True)
            number_classes = logits_list.shape[1]
            train_y = label_list
            pred_class = np.argmax(logits_list, axis=1)

            pred_pdf = {}
            # pdf_entropy = {}
            for i in range(number_classes):
                # print(i)
                # pos_entropy = {}
                top_sequence = {}
                ref_seq = []
                # ids = np.where(train_y == i)[0]
                # print(ids)
                # pred_i = pred_class[ids]
                correct_ids = np.where(pred_class == i)[0] #ids[np.where(pred_class == i)[0]]  ### Correctly predicted sample IDs of class i.
                # print(correct_ids)

                correct_logits = logits_list[correct_ids]
                logits_id = np.argsort(-correct_logits, axis=1)[:, :self.topk]
                # print(logits_id.shape)

                # print(logits)
                # print(logits_id)
                pos_count = 0
                for j in range(self.topk):
                    posterior = {}
                    ij = logits_id[:, j]
                    # print(ij)
                    unique_values, counts = np.unique(ij, return_counts=True)
                    counts = counts / np.sum(counts)

                    # entropy = self.compute_entropy(counts, number_classes)
                    # print("Class "+str(i)+" Position "+str(j)+" Posterior :",unique_values, counts)
                    if True: #entropy < 0.99:
                        # pos_count +=1
                        for z in range(number_classes):
                            if z in unique_values:
                                index = np.where(unique_values == z)[0][0]
                                posterior[unique_values[index]] = counts[index]  #-1/number_classes
                            else:
                                posterior[z] = 0 #-self.reward/number_classes


                    top_sequence[j] = posterior
                    # pos_entropy[j] = entropy
                # print("Class - ",i, pos_count)

                pred_pdf[i] = top_sequence
                # pdf_entropy[i] = pos_entropy
            # print(pred_pdf)
            # file_path = 'cifar100_CLMs_o.pickle'

            # Open the file in binary write mode
            # with open(file_path, 'wb') as file:
            #     Use pickle.dump() to save the dictionary to the file
            #     pickle.dump(pred_pdf, file)
            # print("pickle file created...", flush=True)
            return pred_list, conf_list, label_list, pred_pdf

        elif pca_fit==2:
            number_classes = logits_list.shape[1]
            # print("inside :", pca_obj[0][2])
            pred_pdf_2 = pca_obj
            for i in pred_pdf_2:
                for j in pred_pdf_2[i]:
                    for k in pred_pdf_2[i][j]:
                        if pred_pdf_2[i][j][k] > self.upper / number_classes:  # 5 best
                            pred_pdf_2[i][j][k]  = self.reward/number_classes # 10 best

                        elif pred_pdf_2[i][j][k] > 1 / number_classes:
                            pred_pdf_2[i][j][k] = 1/number_classes #counts[index]

                        elif pred_pdf_2[i][j][k] > 0 :
                            pred_pdf_2[i][j][k] =  -1/number_classes
                        else:
                            pred_pdf_2[i][j][k] = -self.reward/number_classes

            return pred_list, conf_list, label_list, pred_pdf_2



        else:
            return pred_list, conf_list, label_list, None

    # def hook(self, net: nn.Module, input, output):
    #     outputs.append(output)

    def postprocess(self, net: nn.Module, data: Any, pca_obj):
        output = net(data)

        n_cls = int(output.shape[1] /self.topk)
        output_new = output.view(output.shape[0], n_cls, self.topk)
        # print("OUTPUT SHAPE", output_new.shape)
        # print("OUTPUT", np.max(output_new.cpu().numpy(),axis=1))
        # sys.exit()
        score = torch.softmax(output_new[:,:,0], dim=1)
        # score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)


####### Custom OOD Metric "conf" variable is computed within this block #########
        if pca_obj is not None:
            predic_pdf = pca_obj
            original_logits = output_new.cpu().numpy()
            # topk_test = np.argsort(-original_logits, axis=1)[:, :self.topk]

            pred_score_id = self.prediction_strength(original_logits, predic_pdf , self.topk)
            conf = torch.tensor(pred_score_id).cuda()

        # return pred, conf, output
        return pred, conf, output_new[:,:,0]

    def prediction_strength(self,orig_logits, pred_pdf, k):
        a = 0.3
        orig_prob = scipy.special.softmax(orig_logits,axis=1)
        # print(np.sum(orig_prob,axis=1))
        y_pred = np.max(orig_prob,axis=1) #np.argmax(orig_logits,axis=1)
        # print(y_pred)
        # print(y_pred.shape)
        # y_pred = np.argsort(-orig_logits, axis=1)[:, :k]
        # print(y_pred.shape)
        max_logits = np.max(orig_logits[:,:,0], axis=1)
        pred_strength = np.zeros((y_pred.shape[0]))
        pred_cls = np.argmax(orig_prob[:,:,0],axis=1)
        rank_ground_truth = pred_pdf[torch.from_numpy(pred_cls).to('cuda:0')].float()

        # for i in range(orig_prob.shape[0]):
            # print(y_pred[i][:10],pred_cls[i][:10])
        # print(y_pred.shape)
        # print(y_pred,max_logits)
        # pred_strength = a*max_logits + (1-a)*np.sum(y_pred[:,1:],axis=1) #np.zeros((y_pred.shape[0])) #
        # print(pred_strength.shape)
        var =True

##### KL divergence computation ######
        for i in range(y_pred.shape[0]):
            logits_i = torch.from_numpy(orig_logits[i]).to('cuda:0')
            softmax_i = F.softmax(logits_i, dim=0)
            rank_ref_i = rank_ground_truth[i]

            # sys.exit()

            kl_loss = 0.0
            count = 0
            # count_list = []
            for kk in range(1,10):
                # ent_rank_ref = rank_ref_i[:,kk]
                # ent_kk = scipy.stats.entropy(ent_rank_ref.cpu().numpy(),base=self.topk)
                # print(kk,ent_kk)
                # if ent_kk < 0.99: #(1<= kk and kk < 30) or (170 <= kk and kk <200):  #ent_kk < 0.99:
                #     count+=1
                    # count_list.append([kk,ent_kk])
                    # print("yes")
                    # weight = self.compute_entropy(rank_ref_i[:,kk],100)
                log_prob_logits_kk = F.log_softmax(logits_i[:,kk])
                sec_kl_div =  F.kl_div(log_prob_logits_kk, rank_ref_i[:,kk])
                    # if kk==0:
                    #     mul = 5
                    # else:
                    #     mul = 1
                kl_loss += sec_kl_div

            pred_strength[i] = -kl_loss #-(kl_loss/count)  #-kl_loss
            if var == True:
                if pred_cls[i] == 5:
                    if 0.975 < torch.max(softmax_i[:, 0]) < 0.98:
                        print("######################################")
                        # print(logits_i)
                        print(softmax_i)
                        print(rank_ref_i)
                        print("CRAFT_score -",-kl_loss/10)
                        print("MSP score -", torch.max(softmax_i[:, 0]) )
                        # var = False

            # for kk in range(170,200):
            #     log_prob_logits_kk = F.log_softmax(logits_i[:,kk])
            #     sec_kl_div = F.kl_div(log_prob_logits_kk, rank_ref_i[:,kk])
            #     kl_loss += sec_kl_div
            # pred_strength[i] = -kl_loss



        # for i in range(y_pred.shape[0]):
        #     max_lgt = max_logits[i]
        #     # print(max_lgt)
        #     seq = y_pred[i, :]
        #     class_id = seq[0]
        #     score = 0
        #     for position in range(1, k):
        #         score += pred_pdf[class_id][position][seq[position]] #* (1 / pdf_entropy[class_id][position])
        # pred_strength = (1-a)*max_logits + a*pred_strength #(1-self.a)*max_lgt + self.a*score
        return pred_strength
    

    def compute_entropy(self,probabilities,base):
        small_constant = 1e-10  # A small constant to avoid zero probabilities
        modified_probs = probabilities + small_constant
        entropy = -np.round(np.sum(modified_probs * np.log(modified_probs) / np.log(base)), 5)
        # entropy = -np.round(np.sum(modified_probs * np.log10(modified_probs)), 5)
        return entropy

    def set_hyperparam(self,  hyperparam:list):
        self.a = hyperparam[0]
        self.upper = hyperparam[1]
        self.reward = hyperparam[2]

    def get_hyperparam(self):
        return [self.a, self.upper, self.reward]
