"""
This implementation is modified from https://github.com/fungtion/DANN_py3
"""

from tqdm import tqdm
from copy import deepcopy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import IMValidator

import sys
from .buffer import Buffer

class DomainAdversarialNetwork():
    def __init__(self, encoder, classifier, src_train_loader, src_val_loader, device, args):
        self.device = device
        self.set_encoder_classifier(encoder, classifier)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        self.validator = IMValidator()

        self.replay = args.replay
        if self.replay:
            self.buffer = Buffer(args.buffer_size, self.device)
            self.replay_batch_size = args.replay_batch_size
        else:
            self.buffer = None


    def _adapt_train_epoch(self, encoder, classifier, domain_classifier, tgt_train_loader, optimizer, e, epochs, lambda_coeff=0.1, buffer=None, replay_coeff=0.1):
        encoder.train()
        classifier.train()
        domain_classifier.train()

        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)

        total_src_label_loss = 0
        total_src_domain_loss = 0
        total_tgt_domain_loss = 0
        total_replay_loss = 0
        for i in tqdm(range(len_dataloader)):
            p = float(i + e * len_dataloader) / epochs / len_dataloader
            alpha = (2. / (1. + np.exp(-10 * p)) - 1) * lambda_coeff

            src_data, src_label = src_iter.next()
            src_data, src_label = src_data.to(self.device), src_label.to(self.device)

            optimizer.zero_grad()
            src_batch_size = src_data.shape[0]
            domain_label = torch.zeros(src_batch_size).long().to(self.device)

            feature = encoder(src_data)
            class_output = F.log_softmax(classifier(feature), dim=1)
            domain_output = F.log_softmax(domain_classifier(ReverseLayerF.apply(feature, alpha)), dim=1)

            loss_src_label = F.nll_loss(class_output, src_label)
            loss_src_domain = F.nll_loss(domain_output, domain_label)

            tgt_data, _ = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)
            tgt_batch_size = tgt_data.shape[0]
            # print("target batch size", batch_size)
            domain_label = torch.ones(tgt_batch_size).long().to(self.device)

            feature = encoder(tgt_data)
            tgt_class_output = classifier(feature)
            domain_output = F.log_softmax(domain_classifier(ReverseLayerF.apply(feature, alpha)), dim=1)
            loss_tgt_domain = F.nll_loss(domain_output, domain_label)

            replay_loss = torch.tensor(0.)
            if self.replay:
                if not buffer.is_empty():
                    buf_inputs, buf_logits = buffer.get_data(self.replay_batch_size)
                    buf_outputs = classifier(encoder(buf_inputs))
                    replay_loss = F.mse_loss(buf_outputs, buf_logits)
                buffer.add_data(examples=tgt_data, logits=tgt_class_output.data)

            # total_batch_size = src_batch_size + tgt_batch_size
            loss = loss_src_label + loss_src_domain + loss_tgt_domain + replay_loss * replay_coeff
            loss.backward()
            optimizer.step()

            total_src_label_loss += loss_src_label.item()
            total_src_domain_loss += loss_src_domain.item()
            total_tgt_domain_loss += loss_tgt_domain.item()
            total_replay_loss += replay_loss.item()

        total_src_label_loss /= len_dataloader
        total_src_domain_loss /= len_dataloader
        total_tgt_domain_loss /= len_dataloader
        total_replay_loss /= len_dataloader

        return total_src_label_loss, total_src_domain_loss, total_tgt_domain_loss, total_replay_loss

    def _adapt_test_epoch(self, encoder, classifier, domain_classifier, tgt_val_loader, e, epochs, lambda_coeff=0.1):
        encoder.eval()
        classifier.eval()
        domain_classifier.eval()
        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader))
        src_iter = iter(self.src_val_loader)
        tgt_iter = iter(tgt_val_loader)

        total_src_label_loss = 0
        total_src_domain_loss = 0
        total_tgt_domain_loss = 0
        total_domain_correct = 0
        total_data_num = 0
        tgt_logits = []
        for i in tqdm(range(len_dataloader)):
            p = float(i + e * len_dataloader) / epochs / len_dataloader
            alpha = (2. / (1. + np.exp(-10 * p)) - 1) * lambda_coeff
            src_data, src_label = src_iter.next()
            src_data, src_label = src_data.to(self.device), src_label.to(self.device)

            src_batch_size = src_data.shape[0]
            # print("source batch size", batch_size)
            domain_label = torch.zeros(src_batch_size).long().to(self.device)

            feature = encoder(src_data)
            class_output = classifier(feature)
            domain_output = domain_classifier(ReverseLayerF.apply(feature, alpha))

            loss_src_label = F.nll_loss(F.log_softmax(class_output, dim=1), src_label)
            loss_src_domain = F.nll_loss(F.log_softmax(domain_output, dim=1), domain_label)
            domain_pred = domain_output.argmax(dim=1, keepdim=True)
            total_domain_correct += domain_pred.eq(domain_label.view_as(domain_pred)).sum().item()
            total_data_num += src_batch_size

            tgt_data, _ = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)
            tgt_batch_size = tgt_data.shape[0]
            # print("target batch size", batch_size)
            domain_label = torch.ones(tgt_batch_size).long().to(self.device)

            tgt_class_output = classifier(encoder(tgt_data))
            tgt_logits.append(tgt_class_output)
            domain_output = domain_classifier(ReverseLayerF.apply(encoder(tgt_data), alpha))
            loss_tgt_domain = F.nll_loss(F.log_softmax(domain_output, dim=1), domain_label)
            domain_pred = domain_output.argmax(dim=1, keepdim=True)
            total_domain_correct += domain_pred.eq(domain_label.view_as(domain_pred)).sum().item()
            total_data_num += tgt_batch_size

            # total_train_loss += loss.item()
            total_src_label_loss += loss_src_label.item()
            total_src_domain_loss += loss_src_domain.item()
            total_tgt_domain_loss += loss_tgt_domain.item()
            # print('src:', loss_src_domain.item(), 'tgt', loss_tgt_domain.item())

        # total_train_loss /= len_dataloader
        total_src_label_loss /= len_dataloader
        total_src_domain_loss /= len_dataloader
        total_tgt_domain_loss /= len_dataloader
        total_domain_correct /= total_data_num
        tgt_logits = torch.cat(tgt_logits, dim=0)

        return total_src_label_loss, total_src_domain_loss, total_tgt_domain_loss, total_domain_correct, tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, lambda_coeff, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        domain_classifier = DomainClassifier().to(self.device)
        buffer = self.buffer

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()) + list(
                domain_classifier.parameters()),
            lr=args.lr)

        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for e in range(1, args.adapt_epochs + 1):
            train_src_label_loss, train_src_domain_loss, train_tgt_domain_loss, train_replay_loss = self._adapt_train_epoch(encoder, classifier, domain_classifier, tgt_train_loader, optimizer, e, args.adapt_epochs, lambda_coeff, buffer)
            val_src_label_loss, val_src_domain_loss, val_tgt_domain_loss, val_domain_acc, val_tgt_logits = self._adapt_test_epoch(encoder, classifier, domain_classifier, tgt_val_loader, e, args.adapt_epochs, lambda_coeff)


            val_score = self.validator(target_train={'logits': val_tgt_logits})
            print(
                f'Lambda Coef {lambda_coeff} Epoch:{e}/{args.adapt_epochs} Train Source Label Loss:{round(train_src_label_loss, 3)} Train Source Domain Loss: {round(train_src_domain_loss, 3)} Train Target Domain Loss: {round(train_tgt_domain_loss, 3)} \n Val Source Label Loss:{round(val_src_label_loss, 3)} Val Source Domain Loss: {round(val_src_domain_loss, 3)} Val Target Domain Loss: {round(val_tgt_domain_loss, 3)} Val Domain Acc: {round(val_domain_acc, 3)} Val Target Score: {round(val_score, 3)}')

            self.writer.add_scalar('Source Label Loss/train', train_src_label_loss, e)
            self.writer.add_scalar('Source Domain Loss/train', train_src_domain_loss, e)
            self.writer.add_scalar('Target Domain Loss/train', train_tgt_domain_loss, e)
            self.writer.add_scalar('Replay Loss/train', train_replay_loss, e)
            self.writer.add_scalar('Source Label Loss/val', val_src_label_loss, e)
            self.writer.add_scalar('Source Domain Loss/val', val_src_domain_loss, e)
            self.writer.add_scalar('Target Domain Loss/val', val_tgt_domain_loss, e)
            self.writer.add_scalar('Domain Accuracy/val', val_domain_acc, e)
            self.writer.add_scalar('Score/val', val_score, e)

            # val_tgt_label_loss, val_tgt_acc = test_epoch_fn(tgt_encoder, tgt_classifier, device, tgt_val_loader)
            # print(f"Val Target Label Loss: {val_tgt_label_loss} Val Target Acc: {val_tgt_acc}")
            # writer.add_scalar('Target Label Loss/val', val_tgt_label_loss, e)
            # writer.add_scalar('Target Accuracy/val', val_tgt_acc, e)

            if val_score > best_val_score:
                best_val_score = val_score
                best_encoder, best_classifier = deepcopy(encoder), deepcopy(classifier)

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)

        return encoder, classifier, buffer, best_val_score

    def adapt(self, tgt_train_loader, tgt_val_loader, lambda_coeff_list, stage_idx, args):
        val_score_list = []
        encoder_list = []
        classifier_list = []
        buffer_list = []
        revisit = "revisit" if args.revisit else "norevisit"
        replay = "replay" if args.replay else "noreplay"
        for lambda_coeff in lambda_coeff_list:
            run_name = "dann" + str(lambda_coeff).replace(".", "") + "_" + str(args.model_seed) + "_" + replay
            self.writer = SummaryWriter(os.path.join(args.log_dir, revisit, str(stage_idx), run_name))
            encoder, classifier, buffer, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, lambda_coeff, args)
            val_score_list.append(val_score)
            encoder_list.append(encoder)
            classifier_list.append(classifier)
            buffer_list.append(buffer)

        best_idx = max(range(len(val_score_list)), key=val_score_list.__getitem__)
        self.set_encoder_classifier(encoder_list[best_idx], classifier_list[best_idx])
        self.buffer = buffer_list[best_idx]

    def set_encoder_classifier(self, encoder, classifier):
        self.encoder = deepcopy(encoder).to(self.device)
        self.classifier = deepcopy(classifier).to(self.device)

    def get_encoder_classifier(self):
        return self.encoder, self.classifier

class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(9216, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # x = F.log_softmax(x, dim=1)
        return x


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


