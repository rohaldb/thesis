import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class TripletAudio(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self):
        # read in relevant data
        train_data = torch.from_numpy(np.loadtxt('data/trainData.txt', dtype=np.float32))
        test_data = torch.from_numpy(np.loadtxt('data/valData.txt', dtype=np.float32))
        trainKNN = pd.read_csv('data/trainKNN.csv', index_col=0)
        testKNN = pd.read_csv('data/valKNN.csv', index_col=0)

        K = 5 #num KNN we consider positive
        MAX_CLOSE_NEG = 15
        MAX_NEG = 15

        self.train_data = train_data
        self.test_data = test_data
        self.trainKNN = trainKNN
        self.valKNN = valKNN
        self.negIndicies = list(range(K,K + MAX_CLOSE_NEG) + list(-MAX_NEG,-1)) #comibne close and far neg indicies

        if self.train:
            pass # no precomputation needed for train
        else:
            #generate fixed trainin examples
            self.test_triplets = [[
                    index,
                    test_data[testKNN.iloc[index][randint(0, K)]].reshape(-1, 1) #pos
                    test_data[testKNN.iloc[index][np.random.choice(self.negIndicies)]].reshape(-1, 1) #neg
                ] for index in range(0,test_data.shape[0])]


    def __getitem__(self, index):
        if self.train:
            anchor = train_data[trainKNN.iloc[index]]
            pos = train_data[trainKNN.iloc[index][randint(0, K)]].reshape(-1, 1) #pos
            neg = train_data[trainKNN.iloc[index][np.random.choice(self.negIndicies)]].reshape(-1, 1) #neg
        else:
            anch = self.test_data[self.test_triplets[index][0]]
            pos = self.test_data[self.test_triplets[index][1]]
            neg = self.test_data[self.test_triplets[index][2]]

        return (anchor, pos, neg)


    def __len__(self):
        return len(self.test_triplets) + len(self.test_data) # is this right?


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
