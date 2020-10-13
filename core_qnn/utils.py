import torch
from torchvision import transforms
def rgb2quat(rgb):
    gray = torch.mul(rgb, torch.Tensor([[[[0.299]], [[0.587]], [[0.114]]]]))
    # gray = torch.mul(rgb, torch.Tensor([[[[0.]], [[0.]], [[0.]]]]))
    gray = gray[:,0:1,:,:] + gray[:,1:2,:,:] + gray[:,2:3,:,:]
    return torch.cat((gray,rgb),dim=1)

def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    img = transforms.ToPILImage()(img)
    return img


