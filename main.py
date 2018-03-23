import argparse
import os
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from torch.autograd import Variable
from dataset import FashionAI


# Training settings
parser = argparse.ArgumentParser(description='FashionAI')
parser.add_argument('--model', type=str, default='resnet34', metavar='M',
                    help='model name')
parser.add_argument('--attribute', type=str, default='coat_length_labels', metavar='A',
                    help='fashion attribute (default: coat_length_labels)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ci', action='store_true', default=False,
                    help='running CI')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
trainset = FashionAI('./', attribute=args.attribute, split=0.8, ci=args.ci, data_type='train', reset=False)
testset = FashionAI('./', attribute=args.attribute, split=0.8, ci=args.ci, data_type='test', reset=trainset.reset)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(180, 50)
        self.fc2 = nn.Linear(50, FashionAI.AttrKey[args.attribute])

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 180)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if args.ci:
    args.model = 'ci'

if args.model == 'resnet34':
    model = models.resnet34(FashionAI.AttrKey[args.attribute])
else:
    model = Net()

save_folder = os.path.join(os.path.expanduser('.'), 'save', args.attribute, args.model)

if os.path.exists(os.path.join(save_folder, args.model + '_checkpoint.pth')):
    start_epoch = torch.load(os.path.join(save_folder, args.model + '_checkpoint.pth'))
    model.load_state_dict(torch.load(os.path.join(save_folder, args.model + '_' + str(start_epoch) + '.pth')))
else:
    start_epoch = 0

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        #print(target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(model.state_dict(), os.path.join(save_folder, args.model + '_' + str(epoch) + '.pth'))
    torch.save(epoch, os.path.join(save_folder, args.model + '_checkpoint.pth'))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(start_epoch + 1, args.epochs + 1):
    train(epoch)
    test()
