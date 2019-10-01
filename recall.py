import numpy as np
import torch
from collections import defaultdict
from heapq import heappop, heappush

class Recall:
    def __init__(self, num_cand, K):
        self.num_cand = num_cand
        self.K = K

    def calculate(self, data, model, queries, KNN, naive=True):
        self.model = model
        self.data = data

        print('embedding dataset')
        self.outputs = self.model.cpu()(self.data).detach().numpy() #need to call .cpu first so that it doesnt use gpu
        self.code_len = self.outputs.shape[1]
        print("quantizing dataset")
        quantized_outputs = np.heaviside(self.outputs, 0).astype(int)
        self.bucket_hash = BucketHash()
        for i, x in enumerate(quantized_outputs):
            self.bucket_hash.add(x, i) #hash[x].append(i)\
        print('done quantizing into', len(self.bucket_hash.hash.keys()), 'buckets')

        results = []
        for index, q in enumerate(queries):
            pred_knns = self.naive_top_k_items(q) if naive else self.top_k_items(q)
            true_knns = KNN.iloc[index][0:self.K]
            # print(pred_knns)
            results.append(len(set(true_knns).intersection(set(pred_knns)))/len(true_knns)) #comute recall


        return np.array(results)

    def quantization_dist(self, query, bucket):
        proj = self.model(query.unsqueeze(0)).detach().squeeze().numpy()
        code = np.heaviside(proj, 0).astype(int)
        return np.sum(np.logical_xor(code, bucket) * np.linalg.norm(proj))

    #Naive QD ranking
    def naive_top_k_items(self, query):
        cand = [] #candidate set
        print("sorting buckets by QD dist")
        sorted_buckets = sorted(self.bucket_hash.keys(), key=lambda bucket: self.quantization_dist(query, bucket)) #sort by QD
        print("generating candidate set")
        i = 0
        M = self.bucket_hash.keys().shape[0] #num buckets
        while len(cand) < self.num_cand and i < M:
            items = self.bucket_hash.get(sorted_buckets[i])
            cand.extend(items)
            i+=1

        cand = sorted(cand, key=lambda x: np.linalg.norm(self.data[x]-query))
        return cand[0:self.K]

    #Generate-to-Probe QD Ranking
    def top_k_items(self, query):
        self.heap = []
        cand = [] #candidate set
        proj = self.model(query.unsqueeze(0)).detach().squeeze().numpy()
        code = np.heaviside(proj, 0).astype(int)
        i = 0
        sorted_proj, self.proj_map = self.sorted_to_proj_map(proj)

        while len(cand) < self.num_cand and i < 1000:#2**self.code_len:
            b = self.generate_bucket(code, sorted_proj, i)
            items = self.bucket_hash.get(b)
            cand.extend(items)
            i += 1

        cand = sorted(cand, key=lambda x: np.linalg.norm(self.data[x]-query))
        return cand[0:self.K]

    #distance of x from y
    def dist(self, y, x):
        return np.sum(np.multiply(y, abs(x)))

    def generate_bucket(self, code, sorted_proj, i):
        sorted_flip = None
        if i == 0:
            to_add = np.zeros(self.code_len, dtype=int)
            to_add[0] = 1
            heappush(self.heap, (abs(sorted_proj[0]), to_add.tolist()))
            sorted_flip = np.zeros(self.code_len, dtype=int)
        else:
            dist, sorted_flip  = heappop(self.heap)
            sorted_flip = np.asarray(sorted_flip)
            j = np.where(sorted_flip == 1)[0][-1]
            if j < self.code_len-1:
                sorted_flip_plus = np.copy(sorted_flip)
                sorted_flip_plus[j+1] = 1
                to_add = (self.dist(sorted_proj, sorted_flip) + sorted_proj[j+1], sorted_flip_plus.tolist())
                heappush(self.heap, to_add) #convert to list for tiebreaking in queue
                sorted_flip_minus = np.copy(sorted_flip)
                sorted_flip_minus[j] = 0
                sorted_flip_minus[j+1] = 1
                to_add = (self.dist(sorted_proj, sorted_flip) + sorted_proj[j+1] - sorted_proj[j], sorted_flip_minus.tolist())
                heappush(self.heap, to_add)

        return self.sorted_flip_to_bucket(sorted_flip, code)

    #sorts the projection and returns map to original indicies
    def sorted_to_proj_map(self, proj):
        indicies = list(range(0, proj.shape[0]))
        sorted_proj, indicies = zip(*[(p,q) for p,q in sorted(zip(abs(proj),indicies))])
        return sorted_proj, {i: x for i, x in enumerate(indicies)}


    def sorted_flip_to_bucket(self, sorted_flip, code):
        bucket = np.copy(code)
        i = 0
        while i < self.code_len:
            if sorted_flip[i] == 1:
                l = self.proj_map[i]
                bucket[l] = 1 - code[l]
            i += 1
        return bucket


#uses the buckets as keys for a hash. Does so by converting the np arrays to tuples which are hashable
class BucketHash:
    def __init__(self):
        self.hash = defaultdict(list)

    def add(self, bucket, item):
        self.hash[tuple(bucket)].append(item)

    def get(self, bucket):
        return self.hash[tuple(bucket)]

    def keys(self):
        return np.array(list(self.hash.keys()))

    def values(self):
        return np.array(list(self.hash.values()))
