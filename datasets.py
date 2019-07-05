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
        self.neg_indicies = list(range(K+1,K + MAX_CLOSE_NEG)) + list(range(-MAX_FAR_NEG,-1))
        self.train = train

        if self.train:
            self.train_data = torch.from_numpy(np.loadtxt('data/trainData.txt', dtype=np.float32))
            self.train_KNN = pd.read_csv('data/trainKNN.csv', index_col=0)
        else:
            self.test_data = torch.from_numpy(np.loadtxt('data/valData.txt', dtype=np.float32))
            self.test_KNN = pd.read_csv('data/valKNN.csv', index_col=0)
            #generate fixed trainin examples (indicies)
            self.test_triplet_indicies = [[
                    index,
                    self.test_KNN.iloc[index][randint(0, K)], #pos
                    self.test_KNN.iloc[index][np.random.choice(self.neg_indicies)] #neg
                ] for index in range(0, self.test_data.shape[0])]

    def get_dataset(self):
        if self.train:
            return self.train_data
        else:
            return self.test_data

    def __getitem__(self, index):
        if self.train:
            anchor = self.train_data[index]
            pos = self.train_data[self.train_KNN.iloc[index][randint(0, self.K)]]
            neg = self.train_data[self.train_KNN.iloc[index][np.random.choice(self.neg_indicies)]]
        else:
            anchor = self.test_data[self.test_triplet_indicies[index][0]]
            pos = self.test_data[self.test_triplet_indicies[index][1]]
            neg = self.test_data[self.test_triplet_indicies[index][2]]

        # ensure the pos is closer to the point than the neg
        assert( ((anchor-pos)**2).sum() - ((anchor-neg)**2).sum() < 0 )

        return (anchor.reshape(-1, 1), pos.reshape(-1, 1), neg.reshape(-1, 1)), [], index


    def __len__(self):
        return self.train_KNN.shape[0] if self.train else self.test_KNN.shape[0]


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
