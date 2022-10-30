import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_feature = False):
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x1 = self.dropout2(x1)
        output = self.fc2(x1)
        if return_feature:
            return output, x1
        else:
            return output
        # output = F.log_softmax(x, dim=1)  # use nll loss
        # return output


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