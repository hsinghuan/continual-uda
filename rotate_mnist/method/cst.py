"""
Implementation is modified from https://github.com/Liuhong99/CST
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Sequence
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import os
from pytorch_adapt.validators import IMValidator
from .buffer import Buffer

import math

class CycleSelfTrainer():
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

    def _adapt_train_epoch(self, encoder, classifier, tgt_train_loader, ts, optimizer, lr_scheduler, lambda_coeff, ts_coeff=0.08, buffer=None, replay_coeff=0.1):
        encoder.train()
        classifier.train()
        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)

        total_loss = 0
        total_cls_loss = 0
        total_ts_loss = 0
        total_rev_loss = 0
        total_replay_loss = 0

        for i in tqdm(range(len_dataloader)):
            src_data, src_label = src_iter.next()
            src_data, src_label = src_data.to(self.device), src_label.to(self.device)
            tgt_data, _ = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)

            src_y, src_f = classifier(encoder(src_data), return_feature=True)
            tgt_y, tgt_f = classifier(encoder(tgt_data), return_feature=True)

            # generate tgt pseudo labels
            max_prob, pred_u = torch.max(F.softmax(tgt_y), dim=-1)

            # compute cst (reverse)
            target_data_train_r = tgt_f
            target_data_train_r = target_data_train_r / (
                torch.norm(target_data_train_r, dim=-1).reshape(target_data_train_r.shape[0], 1)) # normalize target feature
            target_data_test_r = src_f
            target_data_test_r = target_data_test_r / (torch.norm(target_data_test_r, dim = -1).reshape(target_data_test_r.shape[0], 1)) # normalize source feature
            target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)),
                                        -0.99999999, 0.99999999)
            target_kernel_r = target_gram_r
            test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                                      0.99999999)
            test_kernel_r = test_gram_r
            target_train_label_r = torch.nn.functional.one_hot(pred_u, self.num_cls) - 1 / float(self.num_cls)
            target_test_pred_r = test_kernel_r.mm(
                torch.inverse(target_kernel_r + 0.001 * torch.eye(tgt_y.shape[0]).to(self.device))).mm(target_train_label_r)
            reverse_loss = nn.MSELoss()(target_test_pred_r,
                                        torch.nn.functional.one_hot(src_label, self.num_cls) - 1 / float(self.num_cls))
            cls_loss = F.cross_entropy(src_y, src_label)
            ts_entropy_loss = ts(tgt_y)

            replay_loss = torch.tensor(0.)
            if self.replay:
                if not buffer.is_empty():
                    buf_inputs, buf_logits = buffer.get_data(self.replay_batch_size)
                    buf_outputs = classifier(encoder(buf_inputs))
                    replay_loss = F.mse_loss(buf_outputs, buf_logits)
                buffer.add_data(examples=tgt_data, logits=tgt_y.data)

            loss = cls_loss + ts_entropy_loss * ts_coeff + reverse_loss * lambda_coeff + replay_loss * replay_coeff

            loss.backward()
            optimizer.first_step(zero_grad=True)
            lr_scheduler.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_ts_loss += ts_entropy_loss.item()
            total_rev_loss += reverse_loss.item()
            total_replay_loss += replay_loss.item()


            # prepare for second SGD step
            src_y, src_f = classifier(encoder(src_data), return_feature=True)
            tgt_y, tgt_f = classifier(encoder(tgt_data), return_feature=True)
            max_prob, pred_u = torch.max(F.softmax(tgt_y, dim=1), dim=-1)
            target_data_train_r = tgt_f
            target_data_train_r = target_data_train_r / (
                torch.norm(target_data_train_r, dim=-1).reshape(target_data_train_r.shape[0], 1))
            target_data_test_r = src_f
            target_data_test_r = target_data_test_r / (
                torch.norm(target_data_test_r, dim=-1).reshape(target_data_test_r.shape[0], 1))
            target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)),
                                        -0.99999999, 0.99999999)
            target_kernel_r = target_gram_r
            test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                                      0.99999999)
            test_kernel_r = test_gram_r
            target_train_label_r = torch.nn.functional.one_hot(pred_u, self.num_cls) - 1 / float(self.num_cls)
            target_test_pred_r = test_kernel_r.mm(
                torch.inverse(target_kernel_r + 0.001 * torch.eye(tgt_y.shape[0]).to(self.device))).mm(target_train_label_r)
            reverse_loss = nn.MSELoss()(target_test_pred_r,
                                        torch.nn.functional.one_hot(src_label, self.num_cls) - 1 / float(self.num_cls))

            cls_loss = F.cross_entropy(src_y, src_label)
            ts_entropy_loss = ts(tgt_y)

            replay_loss = torch.tensor(0.)
            if self.replay:
                if not buffer.is_empty():
                    buf_inputs, buf_logits = buffer.get_data(self.replay_batch_size)
                    buf_outputs = classifier(encoder(buf_inputs))
                    replay_loss = F.mse_loss(buf_outputs, buf_logits)
                buffer.add_data(examples=tgt_data, logits=tgt_y.data)

            loss1 = cls_loss + ts_entropy_loss * ts_coeff + reverse_loss * lambda_coeff + replay_loss * replay_coeff
            loss1.backward()
            optimizer.second_step(zero_grad=True)

        total_loss /= len_dataloader
        total_cls_loss /= len_dataloader
        total_ts_loss /= len_dataloader
        total_rev_loss /= len_dataloader
        total_replay_loss /= len_dataloader

        return total_loss, total_cls_loss, total_ts_loss, total_rev_loss, total_replay_loss


    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, tgt_val_loader, ts, lambda_coeff, ts_coeff=0.08):
        encoder.eval()
        classifier.eval()
        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_val_loader)

        total_loss = 0
        total_cls_loss = 0
        total_ts_loss = 0
        total_rev_loss = 0
        tgt_logits = []

        for i in tqdm(range(len_dataloader)):
            src_data, src_label = src_iter.next()
            src_data, src_label = src_data.to(self.device), src_label.to(self.device)
            tgt_data, _ = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)

            src_y, src_f = classifier(encoder(src_data), return_feature=True)
            tgt_y, tgt_f = classifier(encoder(tgt_data), return_feature=True)

            # generate tgt pseudo labels
            max_prob, pred_u = torch.max(F.softmax(tgt_y, dim=1), dim=-1)

            # compute cst (reverse)
            target_data_train_r = tgt_f
            target_data_train_r = target_data_train_r / (
                torch.norm(target_data_train_r, dim=-1).reshape(target_data_train_r.shape[0], 1)) # normalize target feature
            target_data_test_r = src_f
            target_data_test_r = target_data_test_r / (torch.norm(target_data_test_r, dim = -1).reshape(target_data_test_r.shape[0], 1)) # normalize source feature
            target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)),
                                        -0.99999999, 0.99999999)
            target_kernel_r = target_gram_r
            test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                                      0.99999999)
            test_kernel_r = test_gram_r
            target_train_label_r = torch.nn.functional.one_hot(pred_u, self.num_cls) - 1 / float(self.num_cls)
            target_test_pred_r = test_kernel_r.mm(
                torch.inverse(target_kernel_r + 0.001 * torch.eye(tgt_y.shape[0]).to(self.device))).mm(target_train_label_r)
            reverse_loss = nn.MSELoss()(target_test_pred_r,
                                        torch.nn.functional.one_hot(src_label, self.num_cls) - 1 / float(self.num_cls))
            cls_loss = F.cross_entropy(src_y, src_label)
            ts_entropy_loss = ts(tgt_y)
            loss = cls_loss + ts_entropy_loss * ts_coeff + reverse_loss * lambda_coeff

            if math.isnan(loss):
                print("cls loss:", cls_loss)
                print("ts loss:", cls_loss)
                print("reverse loss:", cls_loss)
                print("src f:", src_f)
                print("tgt f:", tgt_f)
                print("src data:", src_data)
                print("tgt data:", tgt_data)
                print("target_data_test_r:", target_data_test_r)
                print("target_test_pred_r:", target_test_pred_r)




            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_ts_loss += ts_entropy_loss.item()
            total_rev_loss += reverse_loss.item()
            tgt_logits.append(tgt_y)

        total_loss /= len_dataloader
        total_cls_loss /= len_dataloader
        total_ts_loss /= len_dataloader
        total_rev_loss /= len_dataloader
        tgt_logits = torch.cat(tgt_logits, dim=0)

        return total_loss, total_cls_loss, total_ts_loss, total_rev_loss, tgt_logits


    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, lambda_coeff, args, test_epoch_fn=None):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        if self.replay:
            buffer = deepcopy(self.buffer)
        else:
            buffer = None
        base_optimizer = SGD
        optimizer = SAM(list(encoder.parameters()) + list(classifier.parameters()), base_optimizer, momentum=0.9, weight_decay=1e-3, lr=args.adapt_lr, adaptive=True, rho=0.5)
        lr_gamma = 0.001
        lr_decay = 0.75
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + lr_gamma* float(x)) ** (-lr_decay))
        ts_loss = TsallisEntropy(temperature=2.0, alpha=1.9)

        best_val_loss = np.inf
        best_val_score = None
        best_encoder, best_classifier = None, None
        patience = 10
        staleness = 0

        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_cls_loss, train_ts_loss, train_rev_loss, train_replay_loss = self._adapt_train_epoch(encoder, classifier, tgt_train_loader, ts_loss, optimizer,
              lr_scheduler, lambda_coeff, buffer=buffer)
            val_loss, val_cls_loss, val_ts_loss, val_rev_loss, tgt_logits = self._adapt_test_epoch(encoder, classifier, tgt_val_loader, ts_loss, lambda_coeff)
            val_score = self.validator(target_train={'logits': tgt_logits})
            # TODO: early-stopping, logging, and printing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)
                best_val_score = val_score
                staleness = 0
            else:
                staleness += 1

            print(
                f'Lambda Coeff {lambda_coeff} Epoch {e}/{self.adapt_epochs} Train Total Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_cls_loss, 3)} Train Ts Loss: {round(train_ts_loss, 3)} Train Rev Loss: {round(train_rev_loss, 3)} Train Replay Loss: {round(train_replay_loss, 3)} \n \
                            Val Total Loss: {round(val_loss, 3)} Val Cls Loss: {round(val_cls_loss, 3)} Val Ts Loss: {round(val_ts_loss, 3)} Val Rev Loss: {round(val_rev_loss, 3)}')

            self.writer.add_scalar('Total Loss/train', train_loss, e)
            self.writer.add_scalar('Total Loss/val', val_loss, e)
            self.writer.add_scalar('Source Label Loss/train', train_cls_loss, e)
            self.writer.add_scalar('Ts Loss/train', train_ts_loss, e)
            self.writer.add_scalar('Rev Loss/train', train_rev_loss, e)
            self.writer.add_scalar('Replay Loss/train', train_replay_loss, e)
            self.writer.add_scalar('Source Label Loss/val', val_cls_loss, e)
            self.writer.add_scalar('Ts Loss/val', val_ts_loss, e)
            self.writer.add_scalar('Rev Loss/val', val_rev_loss, e)
            self.writer.add_scalar('Score/val', val_score, e)


            # test_loss, test_acc = test_epoch_fn(encoder, classifier, self.device, tgt_val_loader)
            # print(f"Lambda Coeff {lambda_coeff} Test Loss: {test_loss} Test Acc: {test_acc}")

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)
        return encoder, classifier, buffer, best_val_score



    def adapt(self, tgt_train_loader, tgt_val_loader, lambda_coeff_list, stage_idx, args, test_epoch_fn=None):
        val_score_list = []
        encoder_list = []
        classifier_list = []
        buffer_list = []
        revisit = "revisit" if args.revisit else "norevisit"
        replay = "replay" if args.replay else "noreplay"
        for lambda_coeff in lambda_coeff_list:
            run_name = "cst" + str(lambda_coeff).replace(".", "") + "_" + str(args.model_seed) + "_" + replay
            self.writer = SummaryWriter(os.path.join(args.log_dir, revisit, str(stage_idx), run_name))
            encoder, classifier, buffer, val_score = self._adapt_train_test(tgt_train_loader, tgt_val_loader, lambda_coeff, args, test_epoch_fn)
            val_score_list.append(val_score)
            encoder_list.append(encoder)
            classifier_list.append(classifier)
            buffer_list.append(buffer)

        best_idx = max(range(len(val_score_list)), key=val_score_list.__getitem__)

        self.set_encoder_classifier(encoder_list[best_idx], classifier_list[best_idx])
        self.buffer = buffer_list[best_idx]
        # return best_encoder, best_classifier


    def set_encoder_classifier(self, encoder, classifier):
        self.encoder = deepcopy(encoder).to(self.device)
        self.classifier = deepcopy(classifier).to(self.device)

    def get_encoder_classifier(self):
        return self.encoder, self.classifier


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


class TsallisEntropy(nn.Module):

    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape

        pred = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)

        sum_dim = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)

        return 1 / (self.alpha - 1) * torch.sum(
            (1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim=-1)))


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm