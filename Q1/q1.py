import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

EPOCHS = 1000
use_cuda = torch.cuda.is_available()

class XorNet(nn.Module):

    def __init__(self):
        super(XorNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)


    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


net = XorNet()
if use_cuda:
    net = net.cuda()


inputs = list(map(lambda s: Variable(torch.Tensor([s])), [[0, 0], [0, 1], [1, 0],[1, 1]]))
targets = list(map(lambda s: Variable(torch.Tensor([s])), [[0], [1], [1], [0]]))

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.01)

# open file for writing
file  = open("Q1_results.txt", "w")

# train
print('training:')
file.write("training:\n")
for n in range(0, EPOCHS):
    for input,target in zip(inputs,targets):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch: %d Loss: %f" % (n,loss))
    file.write("Epoch: %d Loss: %f\n" % (n,loss))
# test
print('testing:')
file.write("testing:\n")
for input in inputs:
    if use_cuda:
        input = input.cuda()
    output = net(input)
    print("input: [%d,%d] output: %f" % (input.data.cpu().numpy()[0][0],input.data.cpu().numpy()[0][1], output.data.cpu().numpy()[0][0]))
    file.write("input: [%d,%d] output: %f\n" % (input.data.cpu().numpy()[0][0],input.data.cpu().numpy()[0][1], output.data.cpu().numpy()[0][0]))

file.close()