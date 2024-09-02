from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import tqdm
from scipy import stats


cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG, self).__init__()
        self.init_channels = 3
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        return out
    


num_classes = {'cifar10': 10, 'cifar100': 100, 'mnist':10, 'imagenet':1000}
datapath = {
    'cifar10': 'C:/Users/see/Documents/GitHub/data/cifar-10-batches-py',
    'cifar100': 'G:/dataset/cifar100',
    'mnist':'G:/dataset/mnist',
    'imagenet': '/gdata/ImageNet2012'
}


def print_args(args):
    print('ARGUMENTS:')
    for arg in vars(args):
        print(f">>> {arg}: {getattr(args, arg)}")

def load_cv_data(data_aug, batch_size, workers, dataset, data_target_dir, is_spiking = False, time_window=100):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        assert False, f"Unknown dataset : {dataset}"

    if data_aug:
        if dataset == 'svhn':
            train_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
        elif dataset == 'imagenet':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(int(224 / 0.875)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
    else:
        if dataset == 'imagenet':
            train_transform = transforms.Compose([
                transforms.Resize(int(224 / 0.875)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(int(224 / 0.875)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                            ])
            test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                            ])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        # if is_spiking:
        #     train_data[0] = (torch.rand(time_window, (3,32,32)) < train_data[0]).float()
        #     test_data[0] = (torch.rand(time_window, (3,32,32)) < test_data[0]).float()

    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
    elif dataset == 'imagenet':
        train_data = datasets.ImageFolder(root=os.path.join(data_target_dir, 'train'),transform=train_transform)
        test_data = datasets.ImageFolder(root=os.path.join(data_target_dir, 'val'),transform=test_transform)
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_dataloader, test_dataloader


class CIFAR(datasets.CIFAR10):
    def __init__(self, root, train=True, mean=0, std=1, is_spiking=False, time_window=100):
        super().__init__(
            root=root, train=train, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
        )
        self.is_spiking = is_spiking
        self.time_window = time_window

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.is_spiking:
            #img = (torch.rand(self.time_window, *img.shape) - 0.5 < img).float()
            img = img.repeat([self.time_window, 1,1,1])
            #pass
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def preprocess_gearbox(path):
    dfs = []
    for dirname, _, filenames in tqdm.tqdm(os.walk(path)):
        for filename in tqdm.tqdm(filenames, leave=False):
            if filename[0] != "h" and filename[0] != "b":
                continue
                
            # the gearbox state is in the filename prefix (h = healthy, b = broken tooth)
            state = filename[0]
            
            # the load is in the filename suffix
            rpm = int(filename.split('.')[0][5:])
            
            # read in the file
            df = pd.read_csv(os.path.join(dirname, filename))
            
            # include the healthy/broken state
            df['state'] = state
            
            # include the load
            df['load'] = rpm
            
            # append to a list
            dfs.append(df)

    # concatenate all the datasets and reset the index
    df = pd.concat(dfs).reset_index().rename(columns={'index':'sample_index'})
    
    sensor_readings = df.melt(
        id_vars=['sample_index','state', 'load'],
        value_vars=['a1','a2','a3','a4'],
        var_name='sensor',
        value_name='reading'
    )

    

    data = {'a1':[], 'a2': [], 'a3':[]}
    labels = []
    for (state,load,sensor),g in sensor_readings.groupby(['state','load','sensor']):
        if sensor =='a4':
            continue
        vals = g.reading.values
        splits = np.split(vals, range(300,vals.shape[0],300))
        for s in splits[:-1]:  # except the last one
            if sensor == 'a1':
                data[sensor].append([
                    float(load),
                    np.mean(s),
                    np.std(s),
                    stats.kurtosis(s),
                    stats.skew(s),
                    stats.moment(s),
                ])
            else:
                data[sensor].append([
                    np.mean(s),
                    np.std(s),
                    stats.kurtosis(s),
                    stats.skew(s),
                    stats.moment(s),
                ])
            if sensor == 'a1':
                labels.append(int(state=='b'))

    
    df_data = pd.DataFrame(data)
    df_data = df_data.apply(lambda x: ','.join(x.astype(str)), axis=1)
    
    
    df_data = df_data.str.split(',', expand=True)
    df_data = df_data.apply(lambda x: x.str.strip('[]'))
    df_data = df_data.astype(float)
    data = df_data.values


    labels = np.array(labels)
    means = data.mean(axis=0)
    stds = data.std(axis=0) +1e-6
    data = (data - means) / stds

    data = pd.DataFrame(data)
    data['label'] = labels
    # Save data and labels as a dataset to csv
    #df_dataset = pd.DataFrame(data, columns=['sensor_a1', 'sensor_a2', 'sensor_a3','load', 'mean', 'std', 'kurt', 'skew', 'moment'])
    #df_dataset['label'] = labels
    #df_dataset.to_csv('dataset.csv', index=False)


    return data, labels

def preprocess_my_gears(influx_client= "", path="", influx=False, synth=False):
    """
    preprocess the gearbox data, loaded from Influx database in broad format
    """
    df = ""
    if influx == True:

        query_api = influx_client.query_api()

        '''query = """ from(bucket: "vibration_data_train_new")
            |> range(start: -30d)
            |> filter(fn: (r) => r["_measurement"] == "vibr_sensor")
            |> filter(fn: (r) => r["_field"] == "sensor_1" or r["_field"] == "sensor_2" or r["_field"] == "sensor_3")
            |> filter(fn: (r) => r["_time"] <= 2024-06-06T07:41:59.891Z)"""'''

        query = """ from(bucket: "vibration_data_test_2")
            |> range(start: -100d)
            |> filter(fn: (r) => r["_measurement"] == "vibr_sensor")
            |> filter(fn: (r) => r["_field"] == "sensor_1" or r["_field"] == "sensor_2" or r["_field"] == "sensor_3")
            """
    
        tables = query_api.query(query, org="some_org")
        results = []
        for table in tables:
            for record in table.records:
                results.append((record.get_field(), record.get_value(), record.values.get("rpm"), record.values.get("label")))
        
        df = pd.DataFrame(results, columns=['sensor_name', 'sensor_val', 'rpm', 'state'])
        
        df['state'] = df['state'].replace({"healthy": "h", "broken": "b"})

        df = df.reset_index().rename(columns={'index':'sample_index'})

        df = pd.read_csv('train_and_convert/merged_data.csv')

        data = {'sensor_1':[], 'sensor_2': [], 'sensor_3':[]}
        labels = []

        for (state,load,sensor),g in df.groupby(['state','rpm','sensor_name']):

            vals = g.sensor_val.values
            splits = np.split(vals, range(500,vals.shape[0],500))
            for s in splits[:-1]:  # except the last one
                s = s.astype(float)
                if sensor == 'sensor_1':
                    data[sensor].append([
                        float(load),
                        np.mean(s),
                        np.std(s),
                        stats.kurtosis(s),
                        stats.skew(s),
                        stats.moment(s),
                    ])
                else:
                    data[sensor].append([
                        np.mean(s),
                        np.std(s),
                        stats.kurtosis(s),
                        stats.skew(s),
                        stats.moment(s),
                    ])
                if sensor == 'sensor_1':
                    labels.append(int(state=='b'))

        df_data = pd.DataFrame(data)
        df_data = df_data.apply(lambda x: ','.join(x.astype(str)), axis=1)
        
        
        df_data = df_data.str.split(',', expand=True)
        df_data = df_data.apply(lambda x: x.str.strip('[]'))
        df_data = df_data.astype(float)
        data = df_data.values

        df_data['target'] = 1


        labels = np.array(labels)

        means = data.mean(axis=0)
        stds = data.std(axis=0) + 1e-6
        data = (data - means) / stds

        # save for faster loading
        df_dataset = pd.DataFrame(data)
        df_dataset['label'] = labels
        df_dataset.to_csv('dataset_test_2.csv', index=False)

        return data, labels, means, stds

    df_data = df['sensor_val']   
    data = df_data.values.astype(float)
    labels = np.array([0 if label == "h" else 1 for label in df['state']])
    means = np.mean(data)
    stds = np.std(data) +1e-6

    data = (data - means) / stds
    
    return data, labels, means, stds
            


class GearboxDataset(Dataset):
    def __init__(self, model_name, path="", transform=None, mode='train', is_spiking=False, time_window=100, b=False, influx=False, data=None, data_version=None, influx_client=""):
        super().__init__()
        self.path = path
        self.influx = influx
        self.influx_client = influx_client
        if data_version == 'kaggle':
            #self.data, self.targets, self.mean, self.std = preprocess_my_gears(influx=influx, influx_client=influx_client, path=self.path)
            self.my_data, self.targets = preprocess_gearbox(self.path)
        else:
            if mode == 'all':
                self.my_data = pd.read_csv('/Users/peterrolfes/workspace/neuromorpher_demonstrator/my_spiking-proof-of-concept/dataset_test.csv')
            elif mode == 'live':
                self.my_data = data
            else:
                self.my_data = pd.read_csv('/Users/peterrolfes/workspace/neuromorpher_demonstrator/my_spiking-proof-of-concept/dataset_100.csv')
        
        self.data_raw = self.my_data.drop(columns=["label"]).to_numpy()
        self.targets = self.my_data["label"].to_numpy()

        indices = np.arange(len(self.data_raw))
        np.random.seed(42)  # Set seed for reproducibility
        np.random.shuffle(indices)
        self.data_raw = self.data_raw[indices]
        self.targets = self.targets[indices]

        self.transform = transform
        self.mode = mode
        self.is_spiking = is_spiking   
        self.time_window = time_window
        self.model_name = model_name
            
        self.data = torch.tensor(self.data_raw).float()
        self.targets = torch.tensor(self.targets).float()

        self.targets = self.targets.to(torch.int64)
        self.targets = torch.nn.functional.one_hot(self.targets, 2)
        self.targets = self.targets.squeeze()
        
        self.train_split = int(len(self.data) * 0.7)
        self.val_split = int(len(self.data) * 0.1)
        self.test_split = int(len(self.data) * 0.2)
       
        if self.mode == 'train':
            self.data = self.data[:self.train_split]
            self.targets = self.targets[:self.train_split]
        elif self.mode == 'val':
            self.data = self.data[self.train_split:self.train_split+self.val_split]
            self.targets = self.targets[self.train_split:self.train_split+self.val_split]
        elif self.mode == 'test':
            self.data = self.data[self.train_split+self.val_split:]
            self.targets = self.targets[self.train_split+self.val_split:]
        elif self.mode == 'all' or self.mode == 'live':
            pass

        if is_spiking:
            self.time_window = time_window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.targets[idx]
        if self.transform:

            sample = self.transform(sample)

        if self.is_spiking:
            if self.model_name == 'conv1d':
                sample = sample.unsqueeze(0)

            sample = sample.repeat([self.time_window,1,1])

            #sample = (torch.rand(self.time_window, *sample.shape) < sample).float()

        else:
            if self.model_name == 'conv1d':
                sample = sample.unsqueeze(0)
        return sample, label





class Ai4iDataset(Dataset):
    def __init__(self, path, transform=None, train=True, is_spiking=False, time_window=100):
        self.df = pd.read_csv(path, header='infer', delimiter=',')
        # self.df = self.df.sample(frac = 1)
        #self.df = pd.read_csv(path, header=None, sep=',')
        #print(self.df.head(5))
        self.convert_dict = {'L': 0., 'M': 1., 'H': 2.}
        self.df = self.df[["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]]
        self.df['Type']=self.df['Type'].apply(self.convert_dict.get)
        mean1 = self.df['Air temperature [K]'].mean()
        std1 = self.df['Air temperature [K]'].std()
        mean2 = self.df['Process temperature [K]'].mean()
        std2 = self.df['Process temperature [K]'].std()
        mean3 = self.df['Rotational speed [rpm]'].mean()
        std3 = self.df['Rotational speed [rpm]'].std()
        mean4 = self.df['Torque [Nm]'].mean()
        std4 = self.df['Torque [Nm]'].std()
        mean5 = self.df['Tool wear [min]'].mean()
        std5 = self.df['Tool wear [min]'].std()
        self.df['Air temperature [K]'] = (self.df['Air temperature [K]'] - mean1) / std1
        self.df['Process temperature [K]'] = (self.df['Process temperature [K]'] - mean2) / std2
        self.df['Rotational speed [rpm]'] = (self.df['Rotational speed [rpm]'] - mean3) / std3
        self.df['Torque [Nm]'] = (self.df['Torque [Nm]'] - mean4) / std4
        self.df['Tool wear [min]'] = (self.df['Tool wear [min]'] - mean5) / std5
        #print(self.df.head(5))

        self.transform = transform
        self.train = train
        self.is_spiking = is_spiking
        split = int(len(self.df) * 0.9)
        self.data = self.df[["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]" ]]
        self.targets = self.df[["Machine failure"]]

        self.data = torch.tensor(self.data.values).float()
        self.targets = torch.tensor(self.targets.values).float()

        print(sum(self.targets[self.targets[:,0] == 1]))

        self.targets = self.targets.to(torch.int64)
        self.targets = torch.nn.functional.one_hot(self.targets, 2)
        self.targets = self.targets.squeeze()

       
        if self.train:
            split = np.arange(0, len(self.data), 1)
            split = split[split % 5 != 0]
            self.data = self.data[split]
            self.targets = self.targets[split]
        else:
            split = np.arange(0, len(self.data), 5)
            self.data = self.data[split]
            self.targets = self.targets[split]
            print(self.targets[:,1].sum())
            #self.data.index = self.data.index - split
            #self.targets.index = self.targets.index - split


        if is_spiking:
            self.time_window = time_window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(self.data.head())
        #print(idx)
        sample, label = self.data[idx], self.targets[idx]
        # print(self.data[idx])
        if self.transform:

            sample = self.transform(sample)

        if self.is_spiking:
            sample = sample.repeat([self.time_window, 1,1,1])
        return sample, label


