from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import urllib
import tarfile
import os
import os.path
import errno
import numpy as np
import csv
import math


def default_loader(path):
    return Image.open(path).convert('RGB')


class FashionAI(Dataset):

    urls = [
        'http://aliyuntianchiresult.cn-hangzhou.oss.aliyun-inc.com/file/race/documents/231649/fashionAI_attributes_test_a_20180222.tar?Expires=1521704427&OSSAccessKeyId=2zep9f8tkzg6ennfl26ciifi&Signature=ASqsT9nTwQag9bwyt0LzkyNQcXo%3D&response-content-disposition=attachment%3B%20',
        'http://aliyuntianchiresult.cn-hangzhou.oss.aliyun-inc.com/file/race/documents/231649/fashionAI_attributes_train_20180222.tar?Expires=1521705583&OSSAccessKeyId=2zep9f8tkzg6ennfl26ciifi&Signature=WOmSxsvHjz5WNh7KQmq9aSw8Ghw%3D&response-content-disposition=attachment%3B%20',
        'http://aliyuntianchiresult.cn-hangzhou.oss.aliyun-inc.com/file/race/documents/231649/warm_up_train_20180201.tar?Expires=1521705626&OSSAccessKeyId=2zep9f8tkzg6ennfl26ciifi&Signature=CRKekykxgyHq8s6NMGVrNeqX5BM%3D&response-content-disposition=attachment%3B%20',
    ]

    base_folder = 'datasets'

    train_folder = 'base'
    warm_folder = 'web'
    rank_folder = 'rank'

    data_folder = 'Images'
    label_folder = 'Annotations'
    rank_label_folder = 'Tests'

    train_label = 'label.csv'
    rank_label = 'question.csv'

    shuffle_file = '_shuffle.npy'
    train_data_file = '_train_data.npy'
    train_label_file = '_train_label.npy'
    test_data_file = '_test_data.npy'
    test_label_file = '_test_label.npy'
    rank_data_file = '_rank_data.npy'
    rank_index_file = '_rank_index.npy'
    ms_file = '_ms.npy'

    AttrKey = {
        'coat_length_labels':8,
        'collar_design_labels':5,
        'lapel_design_labels':5,
        'neck_design_labels':5,
        'neckline_design_labels':10,
        'pant_length_labels':6,
        'skirt_length_labels':6,
        'sleeve_length_labels':9,
    }

    def __init__(self, root, attribute, split=0.8, data_type='train', reset=False, transform=None,
                 target_transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.attribute = attribute
        self.transform = transform
        self.target_transform = target_transform
        self.data_type = data_type
        self.loader = loader
        self.split = split
        self.reset = reset
        self.depth = 3
        self.width = 224
        self.height = 224

        if self.attribute not in self.AttrKey.keys():
            raise RuntimeError('Attribute not found.')
        else:
            self.nb_classes = self.AttrKey[self.attribute]

        if self.split <= 0 or self.split >= 1:
            self.split = 0.8

        self.download()

        label_file = os.path.join(self.root, self.base_folder, self.train_folder, self.label_folder,
                                  self.train_label)
        csvdata = []
        with open(label_file) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[1] == self.attribute:
                    csvdata.append(row)
            f.close()

        shuffle_file = os.path.join(self.root, self.base_folder, self.train_folder, self.label_folder,
                                    self.attribute + self.shuffle_file)
        if not os.path.exists(shuffle_file) or (self.data_type == 'train' and self.reset):
            shuffle = list(range(len(csvdata)))
            np.random.shuffle(shuffle)
            np.save(shuffle_file, shuffle)
            self.reset = True
        else:
            shuffle = np.load(shuffle_file)

        ms_file = os.path.join(self.root, self.base_folder, self.train_folder, self.data_folder,
                               self.attribute + self.ms_file)
        if os.path.exists(ms_file):
            ms = np.load(ms_file)
            self.mean = ms[0]
            self.std = ms[1]
            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
        else:
            self.reset = True

        if self.data_type == 'train':
            data_file = os.path.join(self.root, self.base_folder, self.train_folder, self.data_folder,
                                    self.attribute + self.train_data_file)
            label_file = os.path.join(self.root, self.base_folder, self.train_folder, self.data_folder,
                                    self.attribute + self.train_label_file)

            if not self.reset and os.path.exists(data_file) and os.path.exists(label_file):
                self.train_data = np.load(data_file)
                self.train_labels = np.load(label_file)
                return

            self.train_data = []
            self.train_labels = []

            count = 0
            for row in shuffle[:math.floor(self.split * len(shuffle))]:
                image_file = os.path.join(self.root, self.base_folder, self.train_folder, csvdata[row][0])
                self.train_data.append(np.uint8(np.array(self.loader(image_file).resize((self.width, self.height))).tolist()))
                self.train_labels.append(csvdata[row][2].find('y'))
                count += 1
                #if count == 80:
                    #break

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((count, self.depth, self.width, self.height))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

            #self.train_labels = np.eye(self.nb_classes, dtype=int)[np.array(self.train_labels).reshape(-1)]

            rdata = self.train_data.reshape((count * self.width * self.height, self.depth))
            self.mean = tuple(np.mean(rdata, 0))
            self.std = tuple(np.std(rdata, 0))

            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])

            np.save(data_file, self.train_data)
            np.save(label_file, self.train_labels)
            np.save(ms_file, [self.mean, self.std])

            #print(len(self.train_data.shape))
            #print(self.train_labels)
        elif self.data_type == 'test':
            data_file = os.path.join(self.root, self.base_folder, self.train_folder, self.data_folder,
                                     self.attribute + self.test_data_file)
            label_file = os.path.join(self.root, self.base_folder, self.train_folder, self.data_folder,
                                      self.attribute + self.test_label_file)

            if not self.reset and os.path.exists(data_file) and os.path.exists(label_file):
                self.test_data = np.load(data_file)
                self.test_labels = np.load(label_file)
                return

            self.test_data = []
            self.test_labels = []

            count = 0
            for row in shuffle[math.ceil(self.split * len(shuffle)):]:
                image_file = os.path.join(self.root, self.base_folder, self.train_folder, csvdata[row][0])
                self.test_data.append(np.uint8(np.array(self.loader(image_file).resize((self.width, self.height))).tolist()))
                self.test_labels.append(csvdata[row][2].find('y'))
                count += 1
                #if count == 20:
                    #break

            self.test_data = np.concatenate(self.test_data)
            self.test_data = self.test_data.reshape((count, self.depth, self.width, self.height))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

            #self.test_labels = np.eye(self.nb_classes, dtype=int)[np.array(self.test_labels).reshape(-1)]

            np.save(data_file, self.test_data)
            np.save(label_file, self.test_labels)

            #print(self.test_data.shape)
            #print(self.test_labels)
        elif self.data_type == 'eval':
            data_file = os.path.join(self.root, self.base_folder, self.rank_folder, self.data_folder,
                                     self.attribute + self.rank_data_file)
            index_file = os.path.join(self.root, self.base_folder, self.rank_folder, self.data_folder,
                                     self.attribute + self.rank_index_file)

            if not self.reset and os.path.exists(data_file) and os.path.exists(index_file):
                self.eval_data = np.load(data_file)
                self.eval_index = np.load(index_file)
                return

            self.eval_data = []
            self.eval_index = []
            label_file = os.path.join(self.root, self.base_folder, self.rank_folder, self.rank_label_folder,
                                      self.rank_label)
            count = 0
            with open(label_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[1] == self.attribute:
                        image_file = os.path.join(self.root, self.base_folder, self.rank_folder, row[0])
                        self.eval_data.append(np.uint8(np.array(self.loader(image_file).resize((self.width, self.height))).tolist()))
                        self.eval_index.append(row[0])
                        count += 1
                        #if count == 2:
                            #break
                f.close()

            self.eval_data = np.concatenate(self.eval_data)
            self.eval_data = self.eval_data.reshape((count, self.depth, self.width, self.height))
            self.eval_data = self.eval_data.transpose((0, 2, 3, 1))

            np.save(data_file, self.eval_data)
            np.save(index_file, self.eval_index)

            #print(self.test_data.shape)

    def __getitem__(self, index):
        if self.data_type == 'train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.data_type == 'test':
            img, target = self.test_data[index], self.test_labels[index]
        elif self.data_type == 'eval':
            img, target = self.eval_data[index], self.eval_index[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.data_type == 'train':
            return len(self.train_data)
        elif self.data_type == 'test':
            return len(self.test_data)
        elif self.data_type == 'eval':
            return len(self.eval_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder, self.train_folder)) and \
               os.path.exists(os.path.join(self.root, self.base_folder, self.warm_folder)) and \
               os.path.exists(os.path.join(self.root, self.base_folder, self.rank_folder))

    def download(self):
        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.base_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        def callbackfunc(blocknum, blocksize, totalsize):
            percent = 100.0 * blocknum * blocksize / totalsize
            if percent > 100:
                percent = 100
            print
            "%.2f%%" % percent

        for url in self.urls:
            print('Downloading ' + url)
            filename = url.rpartition('?')[0].rpartition('/')[2]
            file_path = os.path.join(self.root, self.base_folder, filename)
            urllib.request.urlretrieve(url, file_path, callbackfunc)
            with tarfile.open(file_path) as tar_f:
                tar_f.extractall()


if __name__ == "__main__":
    FashionAI('./', 'coat_length_labels')
