import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchsummary, torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import rgb2quat, normalize_output
import matplotlib.pyplot as plt
from model import CIFAR10ConvNet, CIFAR10QConvNet, CIFAR10QConvNetBN, CIFAR10ConvNetBN, CIFAR10QConvNetBN_NormalPool, CIFAR10QConvNet_NormalPool
import numpy as np

test_trainsform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
validation_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_trainsform)
validation_loader   = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=32, shuffle=True)

visualize_dataset   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

# model = CIFAR10QConvNetBN_NormalPool()
# # model.cuda()
# model.load_state_dict(torch.load("C:/Users/QuangDuy/Downloads/norrmal_pool_model.pth", map_location=torch.device('cpu')))
# model.eval()
# # torchsummary.summary(model, (4,32,32))
# model.cpu()
# # print(model.state_dict())

model_amp = CIFAR10QConvNetBN()
# model_amp.cuda()
model_amp.load_state_dict(torch.load("C:/Users/QuangDuy/Downloads/max_amp_pool_model.pth", map_location=torch.device('cpu')))
model_amp.eval()
# torchsummary.summary(model_amp, (4,32,32))
model_amp.cpu()

# Plot some images

for index in range(3):

    data, _ = validation_dataset[index]
    # print(data.shape)
    data = torch.unsqueeze(data, dim=0)
    # print(data.shape)
    data = rgb2quat(data)

    input, _ = visualize_dataset[index]
    input.unsqueeze_(0)
    # print(input.shape)
    # input_ = rgb2quat(input)
    # output = model(input_)
    # print(output.shape)

    img = np.squeeze(input.cpu().detach().numpy(), axis=0)
    img = np.transpose(img, (1,2,0))
    img_ = img[...,::-1].copy()

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model_amp.conv1.register_forward_hook(get_activation('conv1'))
    # print(data.shape)
    output = model_amp(data)
    # print(output.shape)
    act = activation['conv1'].squeeze()
    # print(act.shape)

    quat_components = torch.chunk(act, 4, dim=0)
    r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
    # r.transpose(0,2).transpose(1,2)
    # i.transpose(0,2).transpose(1,2)
    # j.transpose(0,2).transpose(1,2)
    # k.transpose(0,2).transpose(1,2)
    # print(r.shape)
    # num_channel = torch.randint(0,32,(1,))
    num_channel = torch.tensor([23], dtype=torch.long)
    print(num_channel)
    row = 2
    fig, axarr = plt.subplots(row,6)
    
    for ii in range (row):
        # print(r[:,:,num_channel.data+ii].shape)
        axarr[ii,0].imshow(img_)
        axarr[ii,0].axis('off')
        axarr[ii,0].set_title('original')
        axarr[ii,1].imshow(normalize_output(r[num_channel.data+ii]),cmap='gray') 
        axarr[ii,1].axis('off')   
        axarr[ii,1].set_title('r component')
        axarr[ii,2].imshow(normalize_output(i[num_channel.data+ii]),cmap='gray')  
        axarr[ii,2].axis('off')  
        axarr[ii,2].set_title('i component')
        axarr[ii,3].imshow(normalize_output(j[num_channel.data+ii]),cmap='gray')  
        axarr[ii,3].axis('off')  
        axarr[ii,3].set_title('j component')
        axarr[ii,4].imshow(normalize_output(k[num_channel.data+ii]),cmap='gray')   
        axarr[ii,4].axis('off') 
        axarr[ii,4].set_title('k component')

        rgb = torch.cat((i[num_channel.data+ii], j[num_channel.data+ii], k[num_channel.data+ii]), dim=0)
        axarr[ii,5].imshow(normalize_output(rgb))
        axarr[ii,5].axis('off') 
        axarr[ii,5].set_title('rgb')
    # for i in range(1,5):        
    #     axarr[i].imshow(normalize_output(act[i]))
    plt.tight_layout()
    plt.savefig('demo.png')
    plt.show()
    break
