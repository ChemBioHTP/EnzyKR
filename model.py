import pickle
import sys
import timeit
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.metrics import mean_squared_error,r2_score

def bn_conv2d(in_planes, out_planes, kernel_size, dilated):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes,
                                   kernel_size=kernel_size, dilation=dilated,
                                   padding=(dilated * (kernel_size -1)+1)//2),
                        nn.BatchNorm2d(out_planes), nn.ReLU())


def linears(in_planes, out_planes):
    return nn.Sequential(nn.Linear(in_planes, out_planes),
                         nn.BatchNorm1d(out_planes), nn.ReLU())

class enzykr(nn.Module):
    def __init__(self, device, n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output):
        super(enzykr, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([linears(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([bn_conv2d(1, 1, 2*window+1, 1) for _ in
                                    range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.classifier_linear = nn.Linear(2*dim, 2)
        self.regressor_linear = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words, distance = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        distance_vectors = self.embed_distance(distance)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, 4)
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, 4)
        """distance map encoder"""
        distance = self.W_cnn(distance_vectors)
        
        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector,
                                distance_vectors), 1)
        for j in range(3):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        values = self.classifier_linear(cat_vector)
        
        return values

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        # print(predicted_interaction)

        if train:
            #loss = nn.SmoothL1Loss(predicted_interaction, correct_interaction)
            loss = F.mse_loss(predicted_interaction, correct_interaction)
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            return loss, correct_values, predicted_values
        else:
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            # correct_values = np.concatenate(correct_values)
            # predicted_values = np.concatenate(predicted_values)
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            # predicted_values = list(map(lambda x: np.argmax(x), ys))
            # print(correct_values)
            # print(predicted_values)
            # predicted_scores = list(map(lambda x: x[1], ys))
            return correct_values, predicted_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        trainCorrect, trainPredict = [], []
        for data in dataset:
            loss, correct_values, predicted_values = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()

            #correct_values = math.log10(math.pow(2,correct_values))
            #predicted_values = math.log10(math.pow(2,predicted_values))
            trainCorrect.append(correct_values)
            trainPredict.append(predicted_values)
        rmse_train = np.sqrt(mean_squared_error(trainCorrect,trainPredict))
        r2_train = r2_score(trainCorrect,trainPredict)
        spn_r_train = stats.spearmanr(trainCorrect, trainPredict)
        return loss_total, rmse_train, r2_train, spn_r_train


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        testY, testPredict = [], []
        for data in dataset :
            (correct_values, predicted_values) = self.model(data, train=False)
            #correct_values = math.log10(math.pow(2,correct_values))
            #predicted_values = math.log10(math.pow(2,predicted_values))
            SAE += np.abs(predicted_values-correct_values)
            # SAE += sum(np.abs(predicted_values-correct_values))
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY,testPredict))
        r2 = r2_score(testY,testPredict)
        spn_r = stats.spearmanr(testY, testPredict)
        return MAE, rmse, r2, spn_r

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    #(DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
    # lr, lr_decay, decay_interval, weight_decay, iteration,
    # setting) = sys.argv[1:]
    #(dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
    # iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
    #                        decay_interval, iteration])
    #lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])
    radius=3
    ngram=3
    dim=10
    layer_gnn=4
    side=5
    window=11
    layer_cnn=4
    layer_output=3
    lr=1e-3
    lr_decay = 0.5
    decay_interval=10
    weight_decay=1e-6
    iteration=300
    setting = 'train_denov_224'
    # print(type(radius))

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('./data/input/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'regression', torch.FloatTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    # print(n_fingerprint)  # 3958
    # print(n_word)  # 8542
    # 394 and 474 when radius=1 and ngram=2

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    torch.manual_seed(1234)
    model = enzykr().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MAEs = './output/MAEs--' + setting + '.txt'
    file_model = './output/' + setting
    MAEs = ('Epoch\tTime(sec)\tRMSE_train\tR2_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test')
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration+1):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, rmse_train, r2_train , spn_r_train= trainer.train(dataset_train)
        MAE_dev, RMSE_dev, R2_dev, spn_r_dev = tester.test(dataset_dev)
        MAE_test, RMSE_test, R2_test, spn_r = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, rmse_train, r2_train, spn_r_train,  MAE_dev,
                MAE_test, RMSE_dev, R2_dev,  RMSE_test, R2_test, spn_r]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
