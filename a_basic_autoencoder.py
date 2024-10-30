#pip install pytorch_msssim
#pip install opencv-python
#pip install scikit-learn

#https://github.com/lutzroeder/netron/releases/tag/v7.9.8

#__________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import SSIM, MS_SSIM
import os
import cv2
import random
import json

#helper function to convert pytorch tensor to opencv image
def tensor_to_opencv(img : torch.Tensor):
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img*255.0).clip(0, 255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

#helper function to load an image as a tensor
def load_image_as_tensor(img_path, img_size, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size))
    
    img = (img.astype('float32')/255.0)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

#simple autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, img_size : int, latent_dim : int):
        super().__init__()
        self.latent_dim = latent_dim #number of latent dimensions (size of the bottleneck vector)
        
        #input shape of images in {channels, height, width} format
        input_shape = (3, img_size, img_size)
        #total number of pixel values in the input image Channels x Height x Width
        input_size = input_shape[0]*input_shape[1]*input_shape[2]

        #encoder network with 2 fully connected layers (contract input_size -> 512 -> latent_dim)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(), #ReLU activation function, Rectified Linear Unit (max(0, x)) adds non-linearity
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )

        #decoder network with 2 fully connected layers (expand latent_dim -> 512 -> input_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid(), #sigmoid activation to ensure output pixel values are in [0, 1] range
            nn.Unflatten(1, input_shape)
        )

        #randomize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    #apply the encoder network to the input image
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(THIS_FOLDER, 'data/E_Color_Field_Painting_64')
MODEL_FOLDER = "D:/ML/CLASS_DEMOS/autoencoder_models/auto_01"


IMG_SIZE = 64

LATENT_DIM = 32

EPOCHS = 200

SAVE_EVERY = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(MODEL_FOLDER, exist_ok=True)

model = Autoencoder(IMG_SIZE, LATENT_DIM).to(DEVICE)
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)

extensions = ['jpg', 'jpeg', 'png']
images = [f for f in os.listdir(DATA_FOLDER) if f.split('.')[-1].lower() in extensions]

images_to_show = random.sample(images, 4)


for epoch in range(EPOCHS):
    for img_name in images:
        img_path = os.path.join(DATA_FOLDER, img_name)
        img = load_image_as_tensor(img_path, IMG_SIZE, DEVICE)
        
        optimizer.zero_grad()
        output = model(img)
        #loss = (1.0 - ssim_loss(output, img))
        loss = ((img - output)**2).mean()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    with torch.no_grad():
        if (epoch+1) % SAVE_EVERY == 0:        
            torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, f'weights.pth'))
  
            traced_model : torch.ScriptModule = torch.jit.trace(model, img)    
            traced_model.save(os.path.join(MODEL_FOLDER, f'model.pt'))

            model_info = {
                'image_width': IMG_SIZE,
                'image_height': IMG_SIZE,
                'image_channels': 3,
                'latent_dim': LATENT_DIM
                
            }

            with open(os.path.join(MODEL_FOLDER, 'model_info.json'), 'w') as f:
                json.dump(model_info, f)

            
            
        img_pairs = []
        for img_name in images_to_show:
            img_path = os.path.join(DATA_FOLDER, img_name)
            img = load_image_as_tensor(img_path, IMG_SIZE, DEVICE)
            output = model(img)
            img = tensor_to_opencv(img)
            output = tensor_to_opencv(output)

            img_stack = cv2.hconcat([img, output])
            img_pairs.append(img_stack)

        img_grid = cv2.vconcat(img_pairs)
        #set window scaling
        cv2.namedWindow('images', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('images', 2*128, len(img_pairs)*128)
        cv2.imshow('images', img_grid)
        cv2.waitKey(1)