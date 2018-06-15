import os
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from dlc_bci import load

root_dir = os.getcwd()

# Load train, test data
#316x28x50
train_input, train_target = load(root_dir, train=True)
#100x28x50
test_input, test_target = load(root_dir, train=False)

#train_target = train_target.type(FloatTensor)
#test_target = test_target.type(FloatTensor)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

# limit train to 300 for batch gradient
train_input = train_input[:300]
train_target = train_target[:300]

"""
Input: N x 28 x 50
Conv1: N x 32 x 45
pool1: N x 32 x 15
Conv2: N x 48 x 8
pool2: N x 48 x 4
view: N x 1 x 192
fc1: N x 1 x 200
fc2: N x 1 x 2
"""

class Model(nn.Module):
    def __init__(self, nb_hiddens):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(28, 32, kernel_size=6)
        self.conv2 = nn.Conv1d(32, 48, kernel_size=8)
        self.fc1 = nn.Linear(192, nb_hiddens)
        self.fc2 = nn.Linear(nb_hiddens, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 192)))
        x = self.fc2(x)
        return x


def train_model(model, train_input, train_target, batch_size, learning_rate=1e-1, epochs=250):
    criterion = nn.CrossEntropyLoss()
    # Learning rate
    eta = learning_rate
    optimizer = optim.SGD(model.parameters(), lr = 1e-3)

    for e in range(epochs):
        sum_loss = 0

        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, b, batch_size))
            loss = criterion(output, train_target.narrow(0, b, batch_size))
            sum_loss += loss.data

            model.zero_grad()
            loss.backward()

            optimizer.step()

        # print (e, sum_loss)

def compute_nb_errors(model, input, target, batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), batch_size):
        output = model(input.narrow(0, b, batch_size))
        _, predicted_class = torch.max(output.data, 1)

        for i in range(0, batch_size):
            if target.data[b+i] != predicted_class[i]:
                nb_errors += 1

    return nb_errors


batch_size = 10

nb_hiddens = [10, 50, 100, 200, 300]
for i in range(10):
    print('iter ', i)
    for nb_hidden in nb_hiddens:
        model = Model(nb_hidden)

        train_model(model, train_input, train_target, batch_size)
        nb_errors = compute_nb_errors(model, test_input, test_target, batch_size)
        print ('    nb_hidden: {:d} -> error: {:.02f}% {:d} / {:d}'.format(nb_hidden, 100*nb_errors / test_input.size(0), nb_errors, test_input.size(0)))
