#pip install pytorch_msssim
#pip install opencv-python
#pip install scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import SSIM, MS_SSIM
import torchvision
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
import os
import json
from typing import Union
import cv2

normMean = [0.0, 0.0, 0.0] # [0.5, 0.5, 0.5]
normStd =  [1.0, 1.0, 1.0] #[0.25, 0.25, 0.25]



normXForm = transforms.Normalize(normMean, normStd)
normXFormInv = transforms.Normalize([-x / y for x, y in zip(normMean, normStd)], [1.0 / x for x in normStd])

def tensor_to_opencv(img : torch.Tensor):
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img*255.0).clip(0, 255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def load_image_as_tensor(img_path, width, height, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height))
    
    img = (img.astype('float32')/255.0)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

def calcConvolutionShape(h_w, kernel_size=(1,1), stride=(1,1), pad=(0,0), dilation=1):    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

class Autoencoder():
    def __init__(self, folder:str, device:torch.device):
        self.device = device
        
        self.model = torch.jit.load(os.path.join(folder, "model.pt")).to(device)
        self.model.eval()

        with open(os.path.join(folder, "model_info.json"), 'r') as f:
            self.model_info = json.load(f)

        self.image_width = self.model_info['image_width']
        self.image_height = self.model_info['image_height']
        self.image_channels = self.model_info['image_channels']
        self.latent_dim = self.model_info['latent_dim']

    def encode(self, img:Union[torch.Tensor, str]):
        with torch.no_grad():
            if isinstance(img, str):
                img = load_image_as_tensor(img, self.image_width, self.image_height, self.device)
            img_vec = self.model.encoder(img)
            if hasattr(self.model, 'linearToMean'):
                img_vec = self.model.linearToMean(img_vec)
            return img_vec

    def decode(self, img_vec:torch.Tensor):
        with torch.no_grad():
            img = self.model.decoder(img_vec)
            return img

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, imageFolder, image_channels = 3, transform=None):
        self.imageFolder = imageFolder
        self.transform = transform
        self.imageList = os.listdir(imageFolder)
        self.image_channels = image_channels
        
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imageFolder, self.imageList[idx]))
        if self.image_channels == 1:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img

class AutoencoderBase(nn.Module):
    def __init__(self, image_width, image_height, image_channels, latent_dim, device):
        super().__init__()    
        self.encoder : nn.Module = None
        self.decoder : nn.Module = None 

        self.image_width : int = image_width
        self.image_height : int = image_height
        self.image_channels : int = image_channels
        self.latent_dim : int = latent_dim
        self.device : torch.device = device

    def initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        

    def printShapes(self, size, device):
        #create a dummy input of size size (e.g. 64x64)
        dummy_input = torch.rand(1, 3, size[0], size[1])
        dummy_input = dummy_input.to(device)
        #print the shape of each layer in the encoder and decoder using the dummy input
        print("Encoder:")
        for name, layer in self.encoder.named_children():
            dummy_input = layer(dummy_input)
            print(name, "[",layer, "]", ":", dummy_input.shape)

        dummy_input = torch.rand(1, 3, size[0], size[1])
        dummy_input = dummy_input.to(device)
        dummy_input = self.encode(dummy_input)
        print("Decoder:")
        for name, layer in self.decoder.named_children():
            dummy_input = layer(dummy_input)
            print(name, "[",layer, "]", ":", dummy_input.shape)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)

        with torch.no_grad():
            #save weights
            torch.save(self.state_dict(), os.path.join(folder, "weights.pt"))

            #save traced model
            dummy_input = torch.rand(1, self.image_channels, self.image_height, self.image_width).to(self.device)          
            tracedModel = torch.jit.trace(self, dummy_input)
            tracedModel.save(os.path.join(folder, "model.pt"))

        model_info = {
            'image_width': self.image_width,
            'image_height': self.image_height,            
            'image_channels': self.image_channels,
            'latent_dim': self.latent_dim     
        }
        with open(os.path.join(folder, "model_info.json"), 'w') as f:
            json.dump(model_info, f)


class SimpleAutoencoder(AutoencoderBase):
    def __init__(self, image_width, image_height, latent_dim = 50, image_channels = 3, kernelSize = 3, featureChannels = 4, useSSIMLoss = False, device = torch.device('cpu')):
        
        super().__init__(image_width, image_height, image_channels, latent_dim, device)
        
        img_size = (image_height, image_width)

        reducedSize1 = calcConvolutionShape(img_size, (kernelSize,kernelSize), (2,2), (1,1))
        reducedSize2 = calcConvolutionShape(reducedSize1, (kernelSize,kernelSize), (2,2), (1,1))
        reducedSize3 = calcConvolutionShape(reducedSize2, (kernelSize,kernelSize), (2,2), (1,1))


        totalOutputSize = featureChannels*reducedSize3[0]*reducedSize3[1]


        

        self.useSSIMLoss = useSSIMLoss

        if self.useSSIMLoss:
            self.criterion = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
            #criterion = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
        else:
            self.criterion = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernelSize, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernelSize, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, featureChannels, kernelSize, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(totalOutputSize, latent_dim),
            nn.ReLU(True)
        )
            
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, totalOutputSize),
                nn.ReLU(True),
                nn.Unflatten(1, (featureChannels, reducedSize3[0], reducedSize3[1])),
                nn.ConvTranspose2d(featureChannels, 16, kernelSize, stride=2, padding=1, output_padding=1),  
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, kernelSize, stride=2, padding=1, output_padding=1),  
                nn.ReLU(True),
                nn.ConvTranspose2d(8, image_channels, kernelSize, stride=2, padding=1, output_padding=1), 
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def computeLoss(self, x, xHat):
        if self.useSSIMLoss:
            return 1 -  self.criterion(xHat, x)
        else:
            return  self.criterion(xHat, x)
    


#https://blog.paperspace.com/convolutional-autoencoder/
class VGG16Autoencoder(AutoencoderBase):
    def __init__(self, image_width, image_height, latent_dim = 50, image_channels = 3, outputChannels = 16, useSSIMLoss = False, device = torch.device('cpu')):
        super().__init__(image_width, image_height, image_channels, latent_dim, device)

        img_size = (image_height, image_width)
        reducedSize1 = calcConvolutionShape(img_size, (3,3), (2,2), (1,1))
        reducedSize2 = calcConvolutionShape(reducedSize1, (3,3), (2,2), (1,1))

        totalOutputSize = 4*outputChannels*reducedSize2[0]*reducedSize2[1]
        self.useSSIMLoss = useSSIMLoss

        if self.useSSIMLoss:
            self.criterion = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
            #criterion = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
        else:
            self.criterion = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, outputChannels, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(outputChannels, outputChannels, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(outputChannels, 2*outputChannels, 3, padding=1, stride=2), 
            nn.ReLU(True),
            nn.Conv2d(2*outputChannels, 2*outputChannels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2*outputChannels, 4*outputChannels, 3, padding=1, stride=2), 
            nn.ReLU(True),
            nn.Conv2d(4*outputChannels, 4*outputChannels, 3, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(totalOutputSize, latent_dim),
            nn.ReLU(True)
        )
            
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, totalOutputSize),
                nn.ReLU(True),
                nn.Unflatten(1, (4*outputChannels, reducedSize2[0], reducedSize2[1])),
                nn.ConvTranspose2d(4*outputChannels, 4*outputChannels, 3, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(4*outputChannels, 2*outputChannels, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(2*outputChannels, 2*outputChannels, 3, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(2*outputChannels, outputChannels, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(outputChannels, outputChannels, 3, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(outputChannels, image_channels, 3, padding=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def computeLoss(self, x, xHat):
        if self.useSSIMLoss:
            return 1 -  self.criterion(xHat, x)
        else:
            return  self.criterion(xHat, x)
            


#https://avandekleut.github.io/vae/
class VariationalAutoencoder(AutoencoderBase):
    def __init__(self, image_width, image_height, latent_dim = 50, image_channels = 3, outputChannels = 16, device = torch.device('cpu')):
        super().__init__(image_width, image_height, image_channels, latent_dim, device)

        img_size = (image_height, image_width)
        reducedSize1 = calcConvolutionShape(img_size, (3,3), (2,2), (1,1))
        reducedSize2 = calcConvolutionShape(reducedSize1, (3,3), (2,2), (1,1))

        totalOutputSize = 4*outputChannels*reducedSize2[0]*reducedSize2[1]

 
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, outputChannels, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(outputChannels, 2*outputChannels, 3, padding=1, stride=2), 
            nn.ReLU(True),
            nn.Conv2d(2*outputChannels, 4*outputChannels, 3, padding=1, stride=2), 
            nn.ReLU(True),
            nn.Flatten()           
        )

        self.linearToMean = nn.Linear(totalOutputSize, latent_dim)
        self.linearToStd = nn.Linear(totalOutputSize, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
            
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, totalOutputSize),
                nn.Unflatten(1, (4*outputChannels, reducedSize2[0], reducedSize2[1])),
                nn.ReLU(True),                
                nn.ConvTranspose2d(4*outputChannels, 2*outputChannels, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(2*outputChannels, outputChannels, 3, padding=1, stride=2, output_padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(outputChannels, image_channels, 3, padding=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def computeLoss(self, x, xHat):
        return ((x - xHat)**2).sum() + self.kl

    def encode(self, x):
        x = self.encoder(x)
        mu = self.linearToMean(x)
        log_std = self.linearToStd(x)
        sigma = log_std.exp()
        self.kl = self.klLoss(mu, log_std, sigma)
        z = self.N.sample(mu.shape).to(mu.device) * sigma + mu
        return z

    def klLoss(self, mu, log_std, sigma):
        return -0.5*(1 + log_std - mu**2 - sigma**2).sum()
        #return (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()



