import numpy as np
from PIL import Image
from random import randint
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class TripletAudio(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, train, K, MAX_CLOSE_NEG, MAX_FAR_NEG):
        #comibne close and far neg indicies
        self.K = K
        #note randint is inclusive of params
        self.neg_indicies = np.concatenate((np.arange(K,K + MAX_CLOSE_NEG), np.arange(-MAX_FAR_NEG,0)))
        self.train = train

        if self.train:
            self.train_data = torch.from_numpy(np.loadtxt('data/trainData.txt', dtype=np.float32))
            self.train_knn = pd.read_csv('data/trainKNN.csv', index_col=0)
        else:
            self.test_data = torch.from_numpy(np.loadtxt('data/valData.txt', dtype=np.float32))
            self.test_knn = pd.read_csv('data/valKNN.csv', index_col=0)
            #generate fixed trainin examples (indicies)
            self.test_triplet_indicies = [[
                    index,
                    self.test_knn.iloc[index][randint(0, K-1)], #pos
                    self.weighted_neg_sampler(index, train=False)
                ] for index in range(0, self.test_data.shape[0])]

    #picks a strong negative (< MAX_CLOSE_NEG or > MAX_FAR_NEG) datapoint with prob p_strong_neg
    def weighted_neg_sampler(self, row_index, train, p_strong_neg = 0.9):
        knn = self.train_knn if train else self.test_knn
        data = self.train_data if train else self.test_data

        if np.random.choice(np.arange(0,2), p=[1 - p_strong_neg, p_strong_neg]):
            return knn.iloc[row_index][np.random.choice(self.neg_indicies)] #strong neg
        else:
            # weak neg
            neg_index = randint(0, data.shape[0]-1)
            while neg_index in knn.iloc[row_index].values or neg_index == row_index:
                neg_index = randint(0, data.shape[0]-1)
            return neg_index


    def get_dataset(self):
        if self.train:
            return self.train_data
        else:
            return self.test_data

    def get_knn(self):
        if self.train:
            return self.train_knn
        else:
            return self.test_knn

    def __getitem__(self, index):
        if self.train:
            anchor = self.train_data[index]
            pos_index = self.train_knn.iloc[index][randint(0, self.K-1)]
            pos = self.train_data[pos_index]
            neg_index = self.weighted_neg_sampler(index, True)
            neg = self.train_data[neg_index]
        else:
            anchor = self.test_data[self.test_triplet_indicies[index][0]]
            pos = self.test_data[self.test_triplet_indicies[index][1]]
            neg = self.test_data[self.test_triplet_indicies[index][2]]

        # ensure the pos is closer to the point than the neg
        assert((np.linalg.norm(anchor-pos) - np.linalg.norm(anchor-neg)) < 0)

        return (anchor.reshape(-1, 1), pos.reshape(-1, 1), neg.reshape(-1, 1)), [], index


    def __len__(self):
        return self.train_knn.shape[0] if self.train else self.test_knn.shape[0]


class AudioTrainDataset(Dataset):
    """
    Training dataset
    """

    def __init__(self, K):
        #comibne close and far neg indicies
        self.data = torch.from_numpy(np.loadtxt('data/trainData.txt', dtype=np.float32))
        self.KNN = pd.read_csv('data/trainKNN.csv', index_col=0)
        self.K = K

    def __getitem__(self, index):
        return self.data[index].unsqueeze(-1), [], index

    def __len__(self):
        return self.KNN.shape[0]

class AudioTestDataset(Dataset):
    """
    Training dataset
    """

    def __init__(self, K):
        #comibne close and far neg indicies
        self.data = torch.from_numpy(np.loadtxt('data/valData.txt', dtype=np.float32))
        self.KNN = pd.read_csv('data/valKNN.csv', index_col=0)
        self.K = K

    def __getitem__(self, index):
        return self.data[index].unsqueeze(-1), [], index

    def __len__(self):
        return self.KNN.shape[0]

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset):

        self.dataset = dataset
        self.n_anchors = 10
        self.n_pos = 5
        self.n_close_neg = 10
        self.n_far_neg = 10
        self.batch_size = self.n_anchors * (self.n_pos + self.n_close_neg + self.n_far_neg)
        self.anchor_indicies = np.copy(self.dataset.KNN.index.values)
        np.random.shuffle(self.anchor_indicies) #shuffle so we get random anchors
        self.n_dataset = self.dataset.KNN.shape[0]
        self.used_anchors = 0 #counter to move along the anchor indicies list

    def __iter__(self):

        self.used_anchors = 0
        np.random.shuffle(self.anchor_indicies)

        while self.used_anchors + self.batch_size < self.n_dataset:
            batch = set() #set to ensure no duplicates
            # find new anchor points
            batch_anchor_indicies = self.anchor_indicies[self.used_anchors:self.used_anchors + self.n_anchors]
            self.used_anchors += self.n_anchors
            batch.update(batch_anchor_indicies)
            # add some pos and neg points to the batch
            for anchor_index in batch_anchor_indicies:
                row = self.dataset.KNN.iloc[anchor_index]
                batch.update(np.random.choice(row[0:self.dataset.K], self.n_pos, replace=False)) #add pos
                batch.update(np.random.choice(row[self.dataset.K:], self.n_close_neg + self.n_far_neg, replace=False)) #add neg

            #may have added duplicates in process. If so just fill indicies w random
            while len(batch) < self.batch_size:
                batch.add(self.anchor_indicies[self.used_anchors])
                self.used_anchors += 1

            yield list(batch)

        # print(self.used_anchors, self.batch_size , self.n_dataset)

    def __len__(self):
        return self.n_dataset // self.batch_size
