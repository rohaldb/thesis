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



class MutualInfoLoss(nn.Module):

    def __init__(self):
        super(MutualInfoLoss, self).__init__()

    def forward(self, anchor, pair, membership):
        anchor = anchor.numpy()
        pair = pair.numpy()
        membership = membership.numpy()
        b = anchor.shape[1]
        hamming = (b - (anchor * pair).sum(1))/2
        c_xy = np.histogram2d(hamming, membership, [2,2])[0]
        print(c_xy)
        mi = mutual_info_score(None, None, contingency=c_xy)
        return torch.tensor(mi)

#         return F.kl_div(log_p_d, c)
        
#     def forward(self, anchor, positive, negative):
#         b = anchor.shape[1]
#         hamming_pos = (b - (anchor * positive).sum(1))/2
#         hamming_neg = (b - (anchor * negative).sum(1))/2
#         hamming_d = torch.cat((hamming_pos, hamming_neg))
#         print('pos', hamming_pos)
#         print('neg', hamming_neg)
#         print(hamming_d)

#         hist = torch.histc(hamming_d, bins = b, min = 0, max=b)
#         print(hist)
#         p_d = hist/sum(hist)
#         log_p_d = p_d.log().type(torch.FloatTensor)
        
#         c = torch.cat((torch.ones(hamming_pos.shape), torch.zeros(hamming_neg.shape)))
        
# #         return F.kl_div(log_p_d, c)
        
        
