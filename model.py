import torch.nn as nn
import torch.nn.functional as F
import models


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(10, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 6 * 6, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 6 * 6)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def create_model(name, num_classes):
    if name == 'resnet34':
        model = models.resnet34(True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        nn.init.xavier_uniform(model.fc.weight)
        nn.init.constant(model.fc.bias, 0)
    elif name == 'resnet152':
        model = models.resnet152(True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        nn.init.xavier_uniform(model.fc.weight)
        nn.init.constant(model.fc.bias, 0)
    elif name == 'densenet121':
        model = models.densenet121(True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        nn.init.xavier_uniform(model.classifier.weight)
        nn.init.constant(model.classifier.bias, 0)
    elif name == 'vgg11_bn':
        model = models.vgg11_bn(False, num_classes)
    elif name == 'vgg19_bn':
        model = models.vgg19_bn(True)
        model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, num_classes)
        nn.init.xavier_uniform(model.classifier._modules['6'].weight)
        nn.init.constant(model.classifier._modules['6'].bias, 0)
    elif name == 'alexnet':
        model = models.alexnet(True)
        model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, num_classes)
        nn.init.xavier_uniform(model.classifier._modules['6'].weight)
        nn.init.constant(model.classifier._modules['6'].bias, 0)
    else:
        model = Net(num_classes)

    return model
