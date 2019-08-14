import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from datetime import datetime
import argparse

device = torch.device("cuda")

def parse():
    parser = argparse.ArgumentParser(description='2-hidden-layer over-parameterized ReLU network')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'MNIST', 'Fashion'],
                        help='dataset')
    parser.add_argument('--params', type=str, default='100', help='dim or labels')
    parser.add_argument('--output', type=str, default='', help='output file')
    parser.add_argument('--n_epoch', type=int, default=200, help='n_epoch')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    return parser.parse_args()

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

def run(args, h1, h2, N, lr=0.0002, n_epoch=200):

    W = np.array([1 if np.random.rand() > 0.5 else -1 for _ in range(h2)])
    W = torch.from_numpy(W).float().to(device)

    # functions to calculate minimal eigenvalues
    def relu(vec):
        if type(vec) in (float, int, np.float32, np.float64):
            return vec if vec > 0 else 0
        return np.array([relu(v) for v in vec])

    def calcH_min_eig(X):
        n = len(X)
        H = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i][j] = 0.5
                else:
                    H[i][j] = X[i].dot(X[j]) * (0.5 - np.arccos(X[i].dot(X[j])) / (2 * np.pi))
        eig = np.linalg.eig(H)
        return H, eig, min(eig[0])

    def calcH_min_eig_t(X, A, B):
        # X is n*784 matrix, A is h1*784 matrix, B is h2*h1 matrix, n is #samples, w is h2-vec
        n = len(X)
        h2, h1 = B.shape
        w = W.data.cpu().numpy()
        H = np.zeros([n, n])
        uB = [[None for _ in range(h2)] for _ in range(n)]
        uA = [[None for _ in range(h1)] for _ in range(n)]

        for i in range(n):
            yi = relu(A.dot(X[i]))
            for p in range(h2):
                if yi.dot(B[p]) >= 0:
                    uB[i][p] = w[p] * yi
                else:
                    uB[i][p] = np.zeros(h1)
            for r in range(h1):
                if A[r].dot(X[i]) >= 0:
                    uA[i][r] = (w * B[:,r]).T.dot(B.dot(yi) >= 0) * X[i]
                else:
                    uA[i][r] = np.zeros(X.shape[1])

        for i in range(n):
            for j in range(n):
                H[i][j] = sum(uB[i][p].dot(uB[j][p]) for p in range(h2)) + sum(uA[i][r].dot(uA[j][r]) for r in range(h1))

        H /= h1 * h2
        eig = np.linalg.eig(H)
        return H, eig, min(eig[0])


    # data
    if args.dataset != 'synthetic':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if args.dataset == 'MNIST':
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        elif args.dataset == 'Fashion':
            trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

        if len(args.params) != 2:
            raise(TypeError)
        label0, label1 = int(args.params[0]), int(args.params[1])
        ind = []
        n = 0
        while n < N:
            index = np.random.randint(60000)
            if trainset.train_labels[index].item() == label0:
                n += 1
                ind.append(index)

        n = 0
        while n < N:
            index = np.random.randint(60000)
            if trainset.train_labels[index].item() == label1:
                n +=1
                ind.append(index)

        # shuffle
        ind = np.array(ind)
        np.random.shuffle(ind)
        ind = list(ind)

        class realdata(object):
            def __init__(self):
                self.train_data = trainset.train_data[ind].float()
                self.train_data = self.train_data.view(-1, 28*28)
                labels = trainset.train_labels[ind].numpy()
                for i in range(2*N):
                    if labels[i] == label0:
                        labels[i] = 0
                    else:
                        labels[i] = 1
                self.train_labels = torch.from_numpy(labels)

        trainset = realdata()
        for i in range(2*N):
            trainset.train_data[i] = trainset.train_data[i] / float(np.linalg.norm(trainset.train_data[i]))
    else:
        # synthetic data
        def normalize(vec):
            return vec / np.linalg.norm(vec)

        class synthetic(object):
            def __init__(self, n, d):
                self.train_data = torch.from_numpy(np.array([normalize(np.random.normal(size=d)) for _ in range(n)])).float()
                self.train_labels = torch.from_numpy(normalize(np.random.normal(size=n))).float()

        trainset = synthetic(2*N, int(args.params))


    # 3-layer network
    class Net3(nn.Module):
        def __init__(self):
            super(Net3, self).__init__()
            if args.dataset == 'synthetic':
                self.fc1 = nn.Linear(100, h1)
            else:
                self.fc1 = nn.Linear(28*28, h1)
            self.fc2 = nn.Linear(h1, h2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = W.dot(x)
            return x

    net = Net3().to(device)
    criterion = nn.MSELoss()

    # train
    Loss_record = []
    lambda_record = []
    weight_change_record = [[],[]]
    pattern_change_record = [[],[]]
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(n_epoch):

        # extract initialization parameters A0
        if epoch == 0:
            ly_idx = 0
            for p in net.parameters():
                if ly_idx == 0:
                    A0 = p.data.cpu().numpy()
                elif ly_idx == 1:
                    pass
                elif ly_idx == 2:
                    B0 = p.data.cpu().numpy()
                else:
                    pass
                ly_idx += 1
            pattern0 = [[],[]]
            pattern0[0] = [a.dot(x) >= 0 for a in A0 for x in trainset.train_data.numpy()]
            pattern0[1] = [b.dot(relu(A0.dot(x))) >= 0 for b in B0 for x in trainset.train_data.numpy()]

        ############################################################
        # gradient descent for the net
        running_loss = 0.0
        for i, data in enumerate(zip(trainset.train_data, trainset.train_labels)):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        ############################################################

        if abs(running_loss-50.0) < 0.001:
            raise ValueError

        # extract H(t) and calculate lambda_min(H(t))
        ly_idx = 0
        for p in net.parameters():
            if ly_idx == 0:
                A = p.data.cpu().numpy()
            elif ly_idx == 1:
                pass
            elif ly_idx == 2:
                B = p.data.cpu().numpy()
            else:
                pass
            ly_idx += 1
        _, _, lambda_min = calcH_min_eig_t(trainset.train_data.numpy(), A, B)
        if lambda_min <= 0.0:
            raise ValueError
        lambda_record.append(lambda_min)

        # extract activation pattern and calculate pattern change rates
        pattern = [[],[]]
        pattern[0] = [a.dot(x) >= 0 for a in A for x in trainset.train_data.numpy()]
        pattern[1] = [b.dot(relu(A.dot(x))) >= 0 for b in B for x in trainset.train_data.numpy()]

        if epoch == 0:
            A0 = copy.deepcopy(A)
            B0 = copy.deepcopy(B)
            pattern0 = copy.deepcopy(pattern)

        pattern_change_record[0].append(sum([pattern[0][i] != pattern0[0][i]
                                             for i in range(len(pattern0[0]))])/len(pattern0[0]))
        pattern_change_record[1].append(sum([pattern[1][i] != pattern0[1][i]
                                             for i in range(len(pattern0[1]))])/len(pattern0[1]))

        # extract the change of matrix distance
        weight_change_record[0].append(np.linalg.norm(A-A0, ord='fro'))
        weight_change_record[1].append(np.linalg.norm(B-B0, ord='fro'))

        # extract Loss and print
        Loss_record.append(running_loss)
        if (epoch + 1) % 20 == 0:
            print('%d loss: %.8f' % (epoch + 1, running_loss))
            #print(lambda_min)

    print('Finished Training')
    return (weight_change_record, pattern_change_record, lambda_record, Loss_record)

def main():
    args = parse()

    # parameters
    cand = [(100, 5, 50), (500, 5, 50), (1000, 5, 50), (2000, 5, 50), (4000, 5, 50),
            (5, 100, 50), (5, 500, 50), (5, 1000, 50), (5, 2000, 50), (5, 4000, 50)]
    if args.output:
        filename = os.path.join('results', args.output+'.txt')
    else:
        filename = os.path.join('results', args.dataset+args.params+'.txt')
    for h1, h2, N in cand:
        succeed = 0
        for _ in range(100):
            if succeed == 1:
                break
            print('---------------------------', h1, h2, N, '---------------------------')

            a = datetime.now()
            # weight_change_record, pattern_change_record, lambda_record, Loss_record = run(h1, h2, N, lr=0.002)
            try:
                weight_change_record, pattern_change_record, lambda_record, Loss_record = run(args, h1, h2, N, lr=args.lr, n_epoch=args.n_epoch)
                succeed += 1
                with open(filename, 'a+') as f:
                    f.write('h1: %d, h2: %d, N: %d.\n' % (h1, h2, N))
                    f.write("{}\n".format(weight_change_record))
                    f.write("{}\n".format(pattern_change_record))
                    f.write("{}\n".format(lambda_record))
                    f.write("{}\n".format(Loss_record))
                    f.write(';\n')
            except ValueError:
                print('Zero Error here')
            except:
                print('Other Error here')
            b = datetime.now()
            print(b-a, 'seconds')

if __name__ == '__main__':
    main()

