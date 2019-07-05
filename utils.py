from itertools import combinations

import numpy as np
import torch


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


# def similarity_calculator(indicies, KNN):
#     similarity = np.zeros([indicies.shape[0],indicies.shape[0]])
#     for i1, anchor_index in enumerate(indicies):
#         for i2, point_index in enumerate(indicies):
#             if anchor_index == point_index:
#                 similarity[i1][i2] = 1 #this helps follow the structure of FunctionNegativeTripletSelectors body
#                 continue
#             similarity[i1][i2] = 1 if point_index.item() in KNN.iloc[anchor_index.item()][:5].values else 0
#     return similarity

def similarity_calculator(KNN, K):
    size = KNN.shape[0]
    similarity = np.zeros([size, size])
    for anchor_index, row in KNN.iterrows():
        similarity[anchor_index][anchor_index] = 1 #set it to a match with itself
        for i in row[:K].values:
            similarity[anchor_index][i] = 1
    return similarity


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, KNN, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.similarity = similarity_calculator(KNN, 5)

    def get_triplets(self, embeddings, indicies):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()
        triplets = []
        import time

        for index in indicies:
            global_pos_indicies = np.where(self.similarity[index])[0]
            # print(index.item(), global_pos_indicies)
            label_mask = [i.item() in global_pos_indicies for i in indicies]
            label_indices = np.where(label_mask)[0] #get indicies of all entries which are KNN's
            # print('label mask', label_mask, 'label indicies', label_indices)

            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)
            # NOTE THAT ANCHOR_POS WILL DUPLICATE ACCROSS ITTERATIONS
            # print('negative ind', negative_indices)
            # print('anchor pos', anchor_positives)

            if len(anchor_positives) == 0: #if there are no positive points for this embedding
                continue

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]] #dist btw the each anchor pos pair

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                #compute triplet loss between anchor positive pair and all amchor neg pairs
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                # print('loss values', loss_values)
                hard_negative = self.negative_selection_fn(loss_values)
                # print('hardest neg is', hard_negative)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
            #         print('adding tripplet', [anchor_positive[0], anchor_positive[1], hard_negative])
            # print('----------')

        # i think this may fail in the case len(anchor_positives) is always 0
        if len(triplets) == 0:
            print('len triplets was 0')
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)
        # print(triplets)
        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, KNN, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  KNN = KNN,
                                                                                  cpu=cpu)
