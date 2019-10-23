import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score
import numpy as np

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1).pow(0.5) #torch.dist(anchor, positive, 2)
        distance_negative = (anchor - negative).pow(2).sum(1).pow(0.5) #torch.dist(anchor, negative, 2)#
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean() if size_average else losses.sum()
    
#wraps log calls so that log(0) doesnt return inf
def log_wrapper(a):
    return 0 if a == 0 else a.log()

def deriv_triangle(d,l):
        delta = 0.1
        if (l - delta) <= d < l:
            return 1/delta
        elif l <= d <= (l + delta):
            return -1/delta
        else:
            return 0

class MutualInfoLoss(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, anchor, pairs, membership):
        """
        In the forward pass we receive Tensors containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """        
        b = anchor.shape[1]
        hammings = (b - (anchor * pairs).sum(1))/2
        ctx.save_for_backward(anchor, hammings, membership)
        hammings, membership = [x.detach().numpy() for x in [hammings, membership]]
        c_xy = np.histogram2d(hammings, membership, bins=[b+1,2], range=[[0,b], [0,1]])[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return torch.tensor(mi)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        anchor, hammings, membership = ctx.saved_tensors
        b = anchor.shape[1]
        
        #P(C = 1) and P(C = 0)
        p_c_0 = membership.sum().float()/membership.shape[0]
        p_c_1 = 1 - p_c_0
        pos_membership_card = membership.sum().float()
        neg_membership_card = membership.shape[0] - pos_membership_card
        
        pos_hammings = hammings[membership == 1]
        neg_hammings = hammings[membership == 0]
        #need to get hamming dist distributions all inclusive
        hamming_variants = hammings, pos_hammings, neg_hammings
        hamming_hists = [torch.histc(x, bins=b+1, min=0, max=b) for x in hamming_variants]
        p_d, p_d_pos, p_d_neg = [x/x.sum() for x in hamming_hists]
        
        derivative = 0
        for l in range(0,b+1):
            d_i_d_p_pos = p_c_1 * (log_wrapper(p_d_pos[l]) - log_wrapper(p_d[l])) 
            d_p_pos_d_phi = -1./(2*pos_membership_card) * sum([deriv_triangle(x,l) for x in pos_hammings]) * anchor
            d_i_d_p_neg = p_c_0 * (log_wrapper(p_d_neg[l]) - log_wrapper(p_d[l])) 
            d_p_neg_d_phi = -1/(2*neg_membership_card) * sum([deriv_triangle(x,l) for x in neg_hammings]) * anchor
            derivative += d_i_d_p_pos*d_p_pos_d_phi + d_i_d_p_neg * d_p_neg_d_phi

        return derivative, None, None