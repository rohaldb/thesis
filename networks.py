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
        batched_train = train_data.unsqueeze(-1).reshape(-1,train_data.shape[0],1)
        self.biases =  nn.Parameter(anchor_dist(batched_train).mean(0))

    def anchor_dist(self, x):
        batch_size = x.shape[0]
        t = x.reshape(batch_size, 1, -1) #perform transpose within each batch
        return torch.norm(t - self.anchors, 2, 2).unsqueeze(-1) #ensure return shape is batch x output x 1

    def forward(self, x):
        return self.anchor_dist(x) - self.biases

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
