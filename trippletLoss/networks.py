import torch.nn as nn
import torch.nn.functional as F
import torch

class AnchorNet(nn.Module):

    def __init__(self, train_data, INPUT_D, OUTPUT_D, test=False):
        super(AnchorNet, self).__init__()

        #hard code values for testing purposes
        if test == True:
            self.anchors = nn.Parameter(torch.ones(OUTPUT_D, INPUT_D).type(torch.FloatTensor))
            self.biases = torch.ones(OUTPUT_D)
            return


        self.anchors = nn.Parameter(torch.randn(OUTPUT_D, INPUT_D).type(torch.FloatTensor))

        # set biases to be mean of distance between points and anchors
        aggregate = torch.zeros(OUTPUT_D)
        for point in train_data:
            w0 = torch.norm(point.t() - self.anchors, 2, 1)
            aggregate += w0

        self.biases = nn.Parameter((aggregate/train_data.shape[0]).reshape(-1, 1))

    def forward(self, x):
        return torch.norm(x.t() - anchors, 2, 1).reshape(-1, 1) - biases

    def get_embedding(self, x):
        return self.forward(x)

    def extra_repr(self):
        return 'anchors {}, biases {}'.format(self.anchors.shape, self.biases.shape)

class EmbeddingNet(nn.Module):
    def __init__(self, AnchorNet):
        super(EmbeddingNet, self).__init__()

        self.anchor_net = AnchorNet
        self.embedding = nn.Sequential(
            self.anchor_net,
             nn.Sigmoid()
        )

    def forward(self, x):
        return self.embedding(x)

    def get_embedding(self, x):
        return self.forward(x)



class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
