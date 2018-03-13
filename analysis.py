import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='FashionAI Analysis')
parser.add_argument('--model', type=str, default='resnet34', metavar='M', help='model name')
args = parser.parse_args()

log_file = './save/pant_length_labels/resnet34_50.log'

with open(log_file, 'r') as f:
    flogs = f.readlines()

trainset = {
    'loss' : [],
}

testset = {
    'accuracy': [],
    'loss': [],
}

for flog in flogs:
    flog = flog.split()
    if flog[0] == 'Train' and flog[4] == '(0%)]':
        trainset['loss'].append(float(flog[6]))
    elif flog[0] == 'Test':
        temp = flog[6].split('/')
        testset['accuracy'].append(float(temp[0]) / float(temp[1]))
        testset['loss'].append(float(flog[4].split(',')[0]))

epochs = len(trainset['loss'])
x = np.linspace(1, epochs, epochs, endpoint=True)
plt.figure("Analysis")
plt.subplot(311)
plt.plot(x, np.array(trainset['loss']))
plt.subplot(312)
plt.plot(x, np.array(testset['loss']))
plt.subplot(313)
plt.plot(x, np.array(testset['accuracy']))
plt.show()
