"""
The JMMD implementation is from https://github.com/thuml/Transfer-Learning-Library/blob/7f0bc105f1d8adedf6e8d281a29ced8814d96065/tllib/alignment/jan.py#L19
The Gaussian kernel implementation is from https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/kernels.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Sequence
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import os
from pytorch_adapt.validators import IMValidator
from .buffer import Buffer


class JointAdaptationNetwork():
    def __init__(self, encoder, classifier, src_train_loader, src_val_loader, device, args):

        self.device = device
        self.set_encoder_classifier(encoder, classifier)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        thetas = None  # none adversarial
        self.jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                (GaussianKernel(sigma=0.92, track_running_stats=False),)
            ),
            linear=False, thetas=thetas
        ).to(self.device)

        self.adapt_epochs = args.adapt_epochs

        self.validator = IMValidator()

        self.replay = args.replay
        if self.replay:
            self.buffer = Buffer(args.buffer_size, self.device)
            self.replay_batch_size = args.replay_batch_size

    def _adapt_train_epoch(self, encoder, classifier, tgt_train_loader, optimizer, lambda_coeff, buffer=None, replay_coeff=0.1):
        encoder.train()
        classifier.train()
        self.jmmd_loss.train()
        len_dataloader = min(len(self.src_train_loader), len(tgt_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(tgt_train_loader)

        total_loss = 0
        total_cls_loss = 0
        total_transfer_loss = 0
        total_replay_loss = 0
        total_src_data_size = 0

        for i in tqdm(range(len_dataloader)):
            src_data, src_label = src_iter.next()
            src_data, src_label = src_data.to(self.device), src_label.to(self.device)

            tgt_data, _ = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)

            if src_data.shape[0] != tgt_data.shape[0]: # one of the loader is done
                break

            src_y, src_f = classifier(encoder(src_data), return_feature = True)
            tgt_y, tgt_f = classifier(encoder(tgt_data), return_feature = True)
            cls_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_label)
            transfer_loss = self.jmmd_loss((src_f, F.softmax(src_y, dim=1)), (tgt_f, F.softmax(tgt_y, dim=1)))


            replay_loss = torch.tensor(0.)
            if self.replay:
                if not buffer.is_empty():
                    buf_inputs, buf_logits = buffer.get_data(self.replay_batch_size)
                    buf_outputs = classifier(encoder(buf_inputs))
                    replay_loss = F.mse_loss(buf_outputs, buf_logits)
                buffer.add_data(examples=tgt_data, logits=tgt_y.data)
            loss = cls_loss + transfer_loss * lambda_coeff + replay_loss * replay_coeff
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * src_data.size(0)
            total_cls_loss += cls_loss.item() * src_data.size(0)
            total_transfer_loss += transfer_loss.item() * src_data.size(0)
            total_replay_loss += replay_loss.item() * src_data.size(0)
            total_src_data_size += src_data.size(0)

        total_loss /= total_src_data_size
        total_cls_loss /= total_src_data_size
        total_transfer_loss /= total_src_data_size
        total_replay_loss /= total_src_data_size
        return total_loss, total_cls_loss, total_transfer_loss, total_replay_loss

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, tgt_val_loader, lambda_coeff):
        encoder.eval()
        classifier.eval()
        self.jmmd_loss.eval()
        len_dataloader = min(len(self.src_val_loader), len(tgt_val_loader))
        src_iter = iter(self.src_val_loader)
        tgt_iter = iter(tgt_val_loader)

        total_loss = 0
        total_cls_loss = 0
        total_transfer_loss = 0
        total_src_data_size = 0
        tgt_logits = []
        for i in tqdm(range(len_dataloader)):
            src_data, src_label = src_iter.next()
            src_data, src_label = src_data.to(self.device), src_label.to(self.device)

            tgt_data, _ = tgt_iter.next()
            tgt_data = tgt_data.to(self.device)

            if src_data.shape[0] != tgt_data.shape[0]:
                break

            src_y, src_f = classifier(encoder(src_data), return_feature = True)
            tgt_y, tgt_f = classifier(encoder(tgt_data), return_feature = True)
            cls_loss = F.nll_loss(F.log_softmax(src_y, dim=1), src_label)
            transfer_loss = self.jmmd_loss((src_f, F.softmax(src_y, dim=1)), (tgt_f, F.softmax(tgt_y, dim=1)))

            loss = cls_loss + transfer_loss * lambda_coeff


            total_loss += loss.item() * src_data.size(0)
            total_cls_loss += cls_loss.item() * src_data.size(0)
            total_transfer_loss += transfer_loss.item() * src_data.size(0)
            total_src_data_size += src_data.size(0)
            tgt_logits.append(tgt_y)

        total_loss /= total_src_data_size
        total_cls_loss /= total_src_data_size
        total_transfer_loss /= total_src_data_size
        tgt_logits = torch.cat(tgt_logits, dim=0)
        return total_loss, total_cls_loss, total_transfer_loss, tgt_logits

    def _adapt_train_test(self, tgt_train_loader, tgt_val_loader, lambda_coeff, args, test_epoch_fn=None):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        if self.replay:
            buffer = deepcopy(self.buffer)
        else:
            buffer = None
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)

        best_val_loss = np.inf
        best_val_score = None
        best_encoder, best_classifier = None, None
        patience = 10
        staleness = 0

        for e in range(1, self.adapt_epochs + 1):
            total_train_loss, total_train_cls_loss, total_train_transfer_loss, total_train_replay_loss = self._adapt_train_epoch(encoder, classifier, tgt_train_loader, optimizer, lambda_coeff, buffer)
            total_val_loss, total_val_cls_loss, total_val_transfer_loss, tgt_logits = self._adapt_test_epoch(encoder, classifier, tgt_val_loader, lambda_coeff)
            val_score = self.validator(target_train={'logits': tgt_logits})
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)
                best_val_score = val_score
                staleness = 0
            else:
                staleness += 1
            print(
                f'Lambda Coeff {lambda_coeff} Epoch {e}/{self.adapt_epochs} Train Total Loss: {round(total_train_loss, 3)} Train Cls Loss: {round(total_train_cls_loss, 3)} Train Transfer Loss: {round(total_train_transfer_loss, 3)} Train Replay Loss: {round(total_train_replay_loss, 3)} \n \
                Val Total Loss: {round(total_val_loss, 3)} Val Cls Loss: {round(total_val_cls_loss, 3)} Val Transfer Loss: {round(total_val_transfer_loss, 3)}')

            self.writer.add_scalar('Total Loss/train', total_train_loss, e)
            self.writer.add_scalar('Total Loss/val', total_val_loss, e)
            self.writer.add_scalar('Source Label Loss/train', total_train_cls_loss, e)
            self.writer.add_scalar('Transfer Loss/train', total_train_transfer_loss, e)
            self.writer.add_scalar('Replay Loss/train', total_train_replay_loss, e)
            self.writer.add_scalar('Source Label Loss/val', total_val_cls_loss, e)
            self.writer.add_scalar('Transfer Loss/val', total_val_transfer_loss, e)

            test_loss, test_acc = test_epoch_fn(encoder, classifier, self.device, tgt_val_loader)
            print(f"Lambda Coeff {lambda_coeff} Test Loss: {test_loss} Test Acc: {test_acc}")

            if staleness > patience:
                break

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)
        return encoder, classifier, buffer, best_val_score


    def adapt(self, tgt_train_loader, tgt_val_loader, lambda_coeff_list, stage, args, test_epoch_fn=None):
        val_score_list = []
        encoder_list = []
        classifier_list = []
        buffer_list = []
        for lambda_coeff in lambda_coeff_list:
            run_name = "jan" + str(lambda_coeff).replace(".", "") + "_" + str(args.model_seed)
            self.writer = SummaryWriter(os.path.join(args.log_dir, str(stage[0]) + '_' + str(stage[1]), run_name))
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


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix
    Gaussian Kernel k is defined by
    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)
    where :math:`x_1, x_2 \in R^d` are 1-d tensors.
    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`
    .. math::
        K(X)_{i,j} = k(x_i, x_j)
    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.
    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``
    Inputs:
        - X (tensor): input group :math:`X`
    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""
    Args:
        kernels (tuple(tuple(torch.nn.Module))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.
        linear (bool): whether use the linear version of JAN. Default: False
        thetas (list(Theta): use adversarial version JAN if not None. Default: None
    Inputs:
        - z_s (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - z_t (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`
    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar
    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.
    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.
    Examples::
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
        >>> layer2_kernels = (GaussianKernel(1.), )
        >>> loss = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
        >>> # layer1 features from source domain and target domain
        >>> z1_s, z1_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # layer2 features from source domain and target domain
        >>> z2_s, z2_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss((z1_s, z2_s), (z1_t, z2_t))
    """

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = True,
                 thetas: Sequence[nn.Module] = None):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)

        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix