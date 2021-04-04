import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
import pdb

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent


def adentropy(F1, feat, lamda, eta=1.0, s='tar', musk=[]):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    if s == 'src':
        loss_adent = - lamda * torch.mean(torch.sum( out_t1 *(torch.log(out_t1 + 1e-5)) , 1))
    elif s == 'tar':
        loss_adent = lamda * torch.mean(torch.sum( out_t1 *(torch.log(out_t1 + 1e-5)) , 1))
        #loss_adent = lamda * torch.mean(torch.sum(  musk * (out_t1 *(torch.log(out_t1 + 1e-5))) , 1)) # ordinary
    return loss_adent

def adentropy_attention(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(  class_attention(torch.sum( out_t1 *(torch.log(out_t1 + 1e-5)), 1)) )
    return loss_adent

def adentropy_pseudo(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    predicted = torch.max(out_t1, dim=1)
    target = smooth_one_hot(predicted[1], out_t1.shape[1], 0.5)
    loss_adent = lamda * CrossEntropySoft(out_t1 , target) #lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def adentropy_state(F1,feat,lamda,eta=1.0, state='tar'):
    out_t1 = F1(feat, reverse=True, eta=eta, s=state)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

def CrossEntropySoft(predicted, target):
    
    return -(target * torch.log(predicted)).sum(dim=1).mean()