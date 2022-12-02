"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import os
from pytorch_adapt.validators import IMValidator
from .buffer import Buffer

import math

class ClassBalancedSelfTrainer():
    def __init__(self, encoder, classifier, src_train_loader, src_val_loader, device, args):
        self.device = device
        self.set_encoder_classifier(encoder, classifier)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        self.num_cls = 10
        self.adapt_epochs = args.adapt_epochs

        self.validator = IMValidator()

        self.replay = args.replay
        if self.replay:
            self.buffer = Buffer(args.buffer_size, self.device)
            self.replay_batch_size = args.replay_batch_size

    def _adapt_train_epoch(self, encoder, classifier, tgt_train_loader, optimizer, reg_weight, buffer=None, replay_coeff=0.1):
        encoder.train()
        classifier.train()
        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)

        total_src_loss = 0
        total_tgt_loss = 0
        total_replay_loss = 0
        total_src_num = 0
        total_tgt_num = 0
        total_replay_num = 0

        for i in tqdm(range(len_dataloader)):
            src_data, src_label = src_iter.next()
            src_data, src_label = src_data.to(self.device), src_label.to(self.device)
            tgt_data, pseudo_tgt_label = tgt_iter.next()
            tgt_data, pseudo_tgt_label = tgt_data.to(self.device), pseudo_tgt_label.to(self.device)

            src_y = classifier(encoder(src_data))
            tgt_y = classifier(encoder(tgt_data)) # only predict on the confident samples

            src_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_label, reduction='sum')
            tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1), pseudo_tgt_label, reduction='sum') # only train on confident samples

            total_src_num += src_y.shape[0]
            total_tgt_num += tgt_y.shape[0]

            mrkld = torch.sum(- F.log_softmax(tgt_y, dim=1) / self.num_cls)
            tgt_loss += mrkld * reg_weight

            replay_loss = torch.tensor(0.)
            if self.replay:
                if not buffer.is_empty():
                    buf_inputs, buf_logits = buffer.get_data(self.replay_batch_size)
                    buf_outputs = classifier(encoder(buf_inputs))
                    replay_loss = F.mse_loss(buf_outputs, buf_logits, reduction='sum') # changed to sum, if works, replace jan, dann, and cst
                    total_replay_num += buf_logits.shape[0]
                buffer.add_data(examples=tgt_data, logits=tgt_y.data)

            loss = src_loss + tgt_loss + replay_loss * replay_coeff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_replay_loss += replay_loss.item()

        total_loss = (total_src_loss + total_tgt_loss + total_replay_loss) / (total_src_num + total_tgt_num + total_replay_num)
        total_src_loss /= total_src_num
        total_tgt_loss /= total_tgt_num
        if self.replay:
            total_replay_loss /= total_replay_num

        return total_loss, total_src_loss, total_tgt_loss, total_replay_loss


    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, tgt_val_loader, reg_weight):
        encoder.eval()
        classifier.eval()
        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_val_loader)

        total_src_loss = 0
        total_tgt_loss = 0
        total_src_num = 0
        total_tgt_num = 0
        tgt_logits = []

        for i in tqdm(range(len_dataloader)):
            src_data, src_label = src_iter.next()
            src_data, src_label = src_data.to(self.device), src_label.to(self.device)
            tgt_data, pseudo_tgt_label = tgt_iter.next()
            tgt_data, pseudo_tgt_label = tgt_data.to(self.device), pseudo_tgt_label.to(self.device)

            src_y = classifier(encoder(src_data))
            tgt_y = classifier(encoder(tgt_data))  # only predict on the confident samples

            src_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_label, reduction='sum')
            tgt_loss = F.nll_loss(F.log_softmax(tgt_y, dim=1), pseudo_tgt_label,
                                  reduction='sum')  # only train on confident samples

            total_src_num += src_y.shape[0]
            total_tgt_num += tgt_y.shape[0]

            mrkld = torch.sum(- F.log_softmax(tgt_y, dim=1) / self.num_cls)
            tgt_loss += mrkld * reg_weight

            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            tgt_logits.append(tgt_y)

        total_loss = (total_src_loss + total_tgt_loss) / (total_src_num + total_tgt_num)
        total_src_loss /= total_src_num
        total_tgt_loss /= total_tgt_num
        tgt_logits = torch.cat(tgt_logits)

        return total_loss, total_src_loss, total_tgt_loss, tgt_logits


    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, reg_weight, args, p_min=0.2, p_max=0.5, p_inc=0.05, test_epoch_fn=None):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        if self.replay:
            buffer = deepcopy(self.buffer)
        else:
            buffer = None

        p_list = [p_min + i * p_inc for i in range(int((p_max - p_min) // p_inc) + 2)]
        total_e = 0

        for p in p_list:
            data_tensor = []
            pseudo_y_hard_label_tensor = []
            for data, _ in tgt_train_loader:
                data = data.to(self.device)
                pseudo_y_hard_label, pseudo_mask = self.pseudo_label(encoder, classifier, data, p)
                data_tensor.append(data[pseudo_mask])
                pseudo_y_hard_label_tensor.append(pseudo_y_hard_label[pseudo_mask])
            data_tensor = torch.cat(data_tensor, dim=0)
            pseudo_y_hard_label_tensor = torch.cat(pseudo_y_hard_label_tensor, dim=0)
            tgt_pseudo_train_loader = DataLoader(dataset=torch.utils.data.TensorDataset(data_tensor, pseudo_y_hard_label_tensor), batch_size=args.batch_size, shuffle=True)

            # tgt_val_datalist = []
            data_tensor = []
            pseudo_y_hard_label_tensor = []
            for data, _ in tgt_val_loader:
                data = data.to(self.device)
                pseudo_y_hard_label, pseudo_mask = self.pseudo_label(encoder, classifier, data, p)
                data_tensor.append(data[pseudo_mask])
                pseudo_y_hard_label_tensor.append(pseudo_y_hard_label[pseudo_mask])
                # tgt_val_datalist.append((data[pseudo_mask], pseudo_y_hard_label[pseudo_mask]))
            data_tensor = torch.cat(data_tensor, dim=0)
            pseudo_y_hard_label_tensor = torch.cat(pseudo_y_hard_label_tensor, dim=0)
            tgt_pseudo_val_loader = DataLoader(dataset=torch.utils.data.TensorDataset(data_tensor, pseudo_y_hard_label_tensor), batch_size=args.batch_size, shuffle=True)
            optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.adapt_lr)

            best_val_loss = np.inf
            best_val_score = None
            best_encoder, best_classifier = None, None
            patience = 20
            staleness = 0
            for e in range(1, args.adapt_epochs + 1):
                total_e += 1
                train_loss, train_src_loss, train_tgt_loss, train_replay_loss = self._adapt_train_epoch(encoder, classifier, tgt_pseudo_train_loader, optimizer, reg_weight, buffer=buffer)
                val_loss, val_src_loss, val_tgt_loss, tgt_logits = self._adapt_test_epoch(encoder, classifier, tgt_pseudo_val_loader, reg_weight)
                val_score = self.validator(target_train={'logits': tgt_logits})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_encoder = deepcopy(encoder)
                    best_classifier = deepcopy(classifier)
                    best_val_score = val_score
                    staleness = 0
                else:
                    staleness += 1

                print(
                    f'Reg Weight {reg_weight} p: {round(p,2)} Epoch {total_e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} Train Replay Loss: {round(train_replay_loss, 3)} \n \
                                Val Total Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)}')

                self.writer.add_scalar('Total Loss/train', train_loss, e)
                self.writer.add_scalar('Total Loss/val', val_loss, e)
                self.writer.add_scalar('Source Loss/train', train_src_loss, e)
                self.writer.add_scalar('Target Pseudo Loss/train', train_tgt_loss, e)
                self.writer.add_scalar('Replay Loss/train', train_replay_loss, e)
                self.writer.add_scalar('Source Loss/val', val_src_loss, e)
                self.writer.add_scalar('Target Pseudo Loss/val', val_tgt_loss, e)
                self.writer.add_scalar('Score/val', val_score, e)
                if staleness > patience:
                    break

            encoder = deepcopy(best_encoder)
            classifier = deepcopy(best_classifier)

        return encoder, classifier, buffer, best_val_score



    def adapt(self, tgt_train_loader, tgt_val_loader, reg_weight_list, stage_idx, args, test_epoch_fn=None):
        val_score_list = []
        encoder_list = []
        classifier_list = []
        buffer_list = []
        revisit = "revisit" if args.revisit else "norevisit"
        replay = "replay" if args.replay else "noreplay"
        for reg_weight in reg_weight_list:
            run_name = args.method + "_" + str(reg_weight).replace(".", "") + "_" + str(args.model_seed) + "_" + replay
            self.writer = SummaryWriter(os.path.join(args.log_dir, revisit, str(stage_idx), run_name))
            encoder, classifier, buffer, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, reg_weight, args)
            val_score_list.append(val_score)
            encoder_list.append(encoder)
            classifier_list.append(classifier)
            buffer_list.append(buffer)

        best_idx = max(range(len(val_score_list)), key=val_score_list.__getitem__)

        self.set_encoder_classifier(encoder_list[best_idx], classifier_list[best_idx])
        self.buffer = buffer_list[best_idx]
        # return best_encoder, best_classifier

    def pseudo_label(self, encoder, classifier, data, p):
        encoder.eval()
        classifier.eval()
        pseudo_y = classifier(encoder(data))
        pseudo_y = F.softmax(pseudo_y, dim=1)
        pseudo_y_confidence, pseudo_y_hard_label = torch.max(pseudo_y, dim=1)
        pseudo_mask = torch.zeros_like(pseudo_y_hard_label, dtype=torch.bool)
        for cls in range(torch.max(pseudo_y_hard_label) + 1):
            cls_num = (pseudo_y_hard_label==cls).sum().item()
            cls_confidence = pseudo_y_confidence[pseudo_y_hard_label==cls] # the confidence of those predicted as cls
            cls_idx = torch.arange(len(pseudo_y_hard_label))[pseudo_y_hard_label==cls] # the true indices of those predicted as cls
            sorted_confidence_idx = torch.argsort(cls_confidence, descending=True)
            top_p_confidence_cls_idx = cls_idx[sorted_confidence_idx][:int(cls_num * p) + 1]
            # print("sorted confidences:", cls_confidence[sorted_confidence_idx])
            pseudo_mask[top_p_confidence_cls_idx] = True
        return pseudo_y_hard_label, pseudo_mask

    def set_encoder_classifier(self, encoder, classifier):
        self.encoder = deepcopy(encoder).to(self.device)
        self.classifier = deepcopy(classifier).to(self.device)

    def get_encoder_classifier(self):
        return self.encoder, self.classifier



