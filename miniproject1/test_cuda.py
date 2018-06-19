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

if torch.cuda.is_available():
    train_input, train_target = train_input.cuda(), train_target.cuda()
    test_input, test_target = test_input.cuda(), test_target.cuda()

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


"""
Input: N x 28 x 50
Conv1: N x 64 x 40
pool1: N x 64 x 4
view: N x 1 x 256
fc1: N x 1 x nb_hiddens
fc2: N x 1 x 2
"""
class Model2(nn.Module):
    def __init__(self, nb_hiddens):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv1d(28, 64, kernel_size=11)
        self.fc1 = nn.Linear(256, nb_hiddens)
        self.fc2 = nn.Linear(nb_hiddens, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=10, stride=10))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

"""
CNN 3
Input: N x 1 x 28 x 50
Conv1: N x 16 x 24 x 46
pool1: N x 16 x 12 x 23
Conv2: N x 32 x 10 x 20
Conv3: N x 64 x 8 x 18
pool2: N x 64 x 4 x 9
view: N x 1 x 2304
fc1: N x 1 x 2304
fc2: N x 1 x 1000
fc3: N x 1 x 2
"""
class Model3(nn.Module):
    def __init__(self, nb_hiddens):
        super(Model3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 4))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(2304, 2304)
        self.fc2 = nn.Linear(2304, 1000)
        self.fc3 = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 2304)))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

"""
Not Working Yet
"""
def LinearModel(nb_hidden):
    return nn.Sequential(nn.Linear(1400, 2000),
    nn.ReLU(),
    nn.Linear(2000, 2000),
    nn.ReLU(),
    nn.Linear(2000, 128),
    nn.ReLU(),
    nn.Linear(128, 2))

def train_model(model, train_input, train_target, batch_size, learning_rate=1e-1, epochs=150):
    criterion = nn.CrossEntropyLoss()
    # Learning rate
    eta = learning_rate
    optimizer = optim.SGD(model.parameters(), lr = 1e-3)

    train_iter = []
    train_error = []

    step = 0

    for e in range(epochs):
        sum_loss = 0

        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, b, batch_size))
            loss = criterion(output, train_target.narrow(0, b, batch_size))
            sum_loss += loss.data

            model.zero_grad()
            loss.backward()

            optimizer.step()
        train_iter.append(step)
        step += 1
        train_error.append(sum_loss)
        #print (e, sum_loss)
    #plt.plot(train_iter, train_error, 'C1')
    #plt.show()

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

train_input = train_input.view(-1,1,28,50)
test_input = test_input.view(-1,1,28,50)

nb_hiddens = [50]
min = 100
for i in range(20):
    print('iter', i)
    for nb_hidden in nb_hiddens:
        for m in [Model3]:
            model = m(nb_hidden)
            if torch.cuda.is_available():
                model = model.cuda()
            train_model(model, train_input, train_target, batch_size)
            nb_errors = compute_nb_errors(model, test_input, test_target, batch_size)
            if nb_errors < min:
                min = nb_errors
                os.remove('trained_model3.pt')
                torch.save(model.state_dict(), 'trained_model3.pt')
            print ('    nb_hidden: {:d} -> error: {:.02f}% {:d} / {:d}'.format(nb_hidden, 100*nb_errors / test_input.size(0), nb_errors, test_input.size(0)))
    print()

best_model = Model3(50)
best_model.load_state_dict(torch.load('trained_model3.pt'))
nb_errors = compute_nb_errors(model, test_input, test_target, batch_size)
print ('########### BEST MODEL ##############')
print ('nb_hidden: {:d} -> error: {:.02f}% {:d} / {:d}'.format(nb_hidden, 100*nb_errors / test_input.size(0), nb_errors, test_input.size(0)))
