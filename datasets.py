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

    def __init__(self, train, K, MAX_CLOSE_NEG, P_STRONG_NEG):
        #comibne close and far neg indicies
        self.K = K
        #note randint is inclusive of args
        self.neg_indicies = np.arange(K+1,K+1 + MAX_CLOSE_NEG) #recall 0 is the element itself
        self.train = train
        self.p_strong_neg = P_STRONG_NEG
        self.max_close_neg = MAX_CLOSE_NEG

        if self.train:
            self.train_data = torch.from_numpy(np.loadtxt('data/trainData.txt', dtype=np.float32))
            self.train_knn = pd.read_csv('data/trainKNN.csv', index_col=0)
        else:
            self.test_data = torch.from_numpy(np.loadtxt('data/valData.txt', dtype=np.float32))
            self.test_knn = pd.read_csv('data/valKNN.csv', index_col=0)
            #generate fixed training examples (indicies)
            self.test_triplet_indicies = [[
                    index,
                    self.test_knn.iloc[index][randint(1, K)], #pos
                    self.weighted_neg_sampler(index, train=False)
                ] for index in range(0, self.test_data.shape[0])]

    #picks a strong negative (< MAX_CLOSE_NEG) datapoint with prob p_strong_neg
    def weighted_neg_sampler(self, row_index, train):
        knn = self.train_knn if train else self.test_knn
        data = self.train_data if train else self.test_data

        if np.random.choice(np.arange(0,2), p=[1 - self.p_strong_neg, self.p_strong_neg]):
            return knn.iloc[row_index][np.random.choice(self.neg_indicies)] #strong neg
        else:
            # weak neg
            neg_index = randint(0, data.shape[0]-1)
            while neg_index in knn.iloc[row_index][0:K+1+self.max_close_neg].values: #while its in strong negs or pos'
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
            pos_index = self.train_knn.iloc[index][randint(1, self.K)]
            pos = self.train_data[pos_index]
            neg_index = self.weighted_neg_sampler(index, True)
            neg = self.train_data[neg_index]

            #assert the point itself is not included
            assert (index != pos_index and index != neg_index)
        else:
            anchor = self.test_data[self.test_triplet_indicies[index][0]]
            pos = self.test_data[self.test_triplet_indicies[index][1]]
            neg = self.test_data[self.test_triplet_indicies[index][2]]

        # ensure the pos is closer to the point than the neg
        assert (np.linalg.norm(anchor-pos) - np.linalg.norm(anchor-neg)) <= 0, "index {0}, pos {1}, neg {2}".format(index, pos_index, neg_index)


        return (anchor, pos, neg), [], index


    def __len__(self):
        return self.train_knn.shape[0] if self.train else self.test_knn.shape[0]

    

class TrainPair(Dataset):
    """
    Training dataset
    """

    def __init__(self, K, pair_sample_size):
        #comibne close and far neg indicies
        self.data = torch.from_numpy(np.loadtxt('data/trainData.txt', dtype=np.float32))
        self.KNN = pd.read_csv('data/trainKNN.csv', index_col=0)
        self.K = K
        self.pair_sample_size = pair_sample_size

    def __getitem__(self, index):
        pairs = self.data[self.KNN.iloc[index][:self.pair_sample_size]]
        membership = np.append(torch.zeros(self.K), torch.ones(self.pair_sample_size - self.K))
        return self.data[index], pairs, membership

    def __len__(self):
        return self.KNN.shape[0]
    

class TrainPair(Dataset):
    """
    Training dataset
    """

    def __init__(self, K, pair_sample_size):
        #comibne close and far neg indicies
        self.data = torch.from_numpy(np.loadtxt('data/trainData.txt', dtype=np.float32))
        self.KNN = pd.read_csv('data/trainKNN.csv', index_col=0)
        self.K = K
        self.pair_sample_size = pair_sample_size

    def __getitem__(self, index):
        pairs = self.data[self.KNN.iloc[index][:self.pair_sample_size]]
        membership = np.append(torch.ones(self.K), torch.zeros(self.pair_sample_size - self.K))
        return self.data[index], pairs, membership
        

    def __len__(self):
        return self.KNN.shape[0]    
    
class TestPair(Dataset):
    """
    Training dataset
    """

    def __init__(self, K, pair_sample_size):
        #comibne close and far neg indicies
        self.data = torch.from_numpy(np.loadtxt('data/valData.txt', dtype=np.float32))
        self.KNN = pd.read_csv('data/valKNN.csv', index_col=0)
        self.K = K
        self.pair_sample_size = pair_sample_size

    def __getitem__(self, index):
        pairs = self.data[self.KNN.iloc[index][:self.pair_sample_size]]
        membership = np.append(torch.ones(self.K), torch.zeros(self.pair_sample_size - self.K))
        return self.data[index], pairs, membership

    def __len__(self):
        return self.KNN.shape[0]