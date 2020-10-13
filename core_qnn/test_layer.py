import torch
from quaternion_layers import QuaternionBatchNorm2d, QuaternionMaxAmpPool2d
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transforms
from quaternion_ops import *
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num):
        super(Model, self).__init__()
        self.BN = QuaternionBatchNorm2d(num)
    def forward(self, x):
        return self.BN(x)

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.pool = QuaternionMaxAmpPool2d(2,2)
    def forward(self, x):
        return self.pool(x)

class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        return image, label
        
    def __len__(self):
        return len(self.images)



def get_var(x):
    quat_components = torch.chunk(x, 4, dim=1)
    r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
    mean_r = torch.mean(r)
    mean_i = torch.mean(i)
    mean_j = torch.mean(j)
    mean_k = torch.mean(k)

    delta_r = r - mean_r
    delta_i = i - mean_i
    delta_j = j - mean_j
    delta_k = k - mean_k

    quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2),axis = (0,2,3))
    return quat_variance

if __name__ == "__main__": 
    """
    print('input check----------------------------')
    num = 4
    x = torch.randn(16, num, 32, 32) * 1.0
    label = torch.randn(16, num, 32, 32) * 1.0
    x_variance = get_var(x)
    # test batch norm
    
    print('mean x: ', torch.mean(x,axis=(0,2,3)))
    print('variance x: ',x_variance)

    model = Model(num)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # print('output check---------------------------')
    print(dict(model.named_buffers()))
    y = model(x)
    y_variance = get_var(y)
    
    # print(y.shape)
    print('mean y: ', torch.mean(y, axis=(0,2,3)))
    print('variance y: ', y_variance)
    # print('input shape ', x.shape)
    # print('output shape ', y.shape) 
    print('---------------------------------------')
    # print('test done ')
    print(dict(model.named_buffers()))

    model.eval()
    y_pre = model(x)
    print(dict(model.named_buffers()))
    """
    
    # Create random images
    images = [x for x in torch.randn(64, 16, 8, 8)*100]
    labels = [x for x in torch.randn(64, 16, 8, 8)*1000]
    dataset = MyDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16)
    criterion = nn.MSELoss()
    model = Model(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    model.train()
    iteration = 0
    for epoch in range(10):
        # print("[Epoch {}]".format(epoch))
        for batch_idx, data in enumerate(loader):
            print("Iteration: {}".format(iteration))
            x, y = data
            optimizer.zero_grad()
            y_predict = model(x)
            loss = criterion(y, y_predict)
            loss.backward()
            optimizer.step()
            print(dict(model.named_buffers()))
            iteration+=1


    # #visualize
    # import matplotlib.pyplot as plt

    # plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    # # Plot Histogram on x
    # # x = np.random.normal(size = 1000)
    # plt.hist(torch.mean(x,axis=(1,2,3)), bins=50)
    # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    # plt.show()

    
    #test max amp pooling
    """
    x = torch.randn(1, 4, 2, 2)
    model = Pool()

    y = model(x)
    print('input shape ', x.shape)
    print('output shape ', y.shape) 

    quat_components = torch.chunk(x, 4, dim=1)
    r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]

    amp = get_modulus(x, vector_form=True)
    print('max amp shape ', amp.shape)

    print('input: ', x)
    print('max amp: ', amp)
    print('output: ', y)
    """