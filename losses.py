import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, rho, beta):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.rho = rho
        self.beta = beta

    def forward(self, anchor, positive, negative, model, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        

        anchors = model.embedding_net.anchor_net.anchors
        #compute l2dist between all pairs of anchors
        anchor_dists = torch.cdist(anchors, anchors)
        t = (self.beta * F.relu(self.rho - anchor_dists))
        regularization = t.sum() - torch.diag(t).sum() #need to exclude diags from sum
        return regularization + losses.mean() if size_average else losses.sum()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, indicies):
        triplets = self.triplet_selector.get_triplets(embeddings, indicies)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
