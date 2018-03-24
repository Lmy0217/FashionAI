import argparse
import os.path
import torch
import numpy as np
import model as m
from torch.autograd import Variable
from dataset import FashionAI
import csv


parser = argparse.ArgumentParser(description='FashionAI Evaluate')
parser.add_argument('--model', type=str, default='resnet34', metavar='M',
                    help='model name')
parser.add_argument('--attribute', type=str, default='coat_length_labels', metavar='A',
                    help='fashion attribute (default: coat_length_labels)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
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
evalset = FashionAI('./', attribute=args.attribute, ci=args.ci, data_type='eval', reset=False)
eval_loader = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size, shuffle=True, **kwargs)

if args.ci:
    args.model = 'ci'

model = m.create_model(args.model, FashionAI.AttrKey[args.attribute])

save_folder = os.path.join(os.path.expanduser('.'), 'save', args.attribute, args.model)

if os.path.exists(os.path.join(save_folder, args.model + '_checkpoint.pth')):
    start_epoch = torch.load(os.path.join(save_folder, args.model + '_checkpoint.pth'))
    model.load_state_dict(torch.load(os.path.join(save_folder, args.model + '_' + str(start_epoch) + '.pth')))
else:
    start_epoch = 0

if args.cuda:
    model.cuda()


def eval():
    model.eval()
    writedata = []
    for data, target in eval_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        output = np.exp(output.cpu().data.numpy()).tolist()
        writedata.extend([ [j, args.attribute, ";".join([ str(ii) for ii in i ])] for (i, j) in zip(output, target) ])

    return writedata


eval_file = os.path.join(os.path.expanduser('.'), 'save', args.attribute, args.model + '_' + str(start_epoch) + '_eval.csv')
with open(eval_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(eval())