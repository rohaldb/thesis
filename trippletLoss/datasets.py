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

        return (anchor.reshape(-1, 1), pos.reshape(-1, 1), neg.reshape(-1, 1))


    def __len__(self):
        return self.train_KNN.shape[0] if self.train else self.test_KNN.shape[0]



class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
