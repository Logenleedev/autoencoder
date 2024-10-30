#pip install pytorch_msssim
#pip install opencv-python
#pip install scikit-learn

import torch
import torch.optim
from torchvision import transforms
from autoencoder import *
import numpy as np
import os
import random
import cv2

#____________________________________________set the parameters for the model training
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#data folder with images to use for training. 
DATA_FOLDER = os.path.join(THIS_FOLDER, 'data/modigliani_128')
#folder to save the trained model
MODEL_FOLDER = "/Users/mac/Desktop/School/Harvard/08_autoencoder_scripts/model"

IMG_SIZE = 128      #image size. USe smaller size for faster training
LATENT_DIM = 120    #size of the latent space vector.
EPOCHS = 200        #number of epochs to train the model
SAVE_EVERY = 10     #save the model every N epochs
RANDOMIZE = False   #add random noise to the images while training to make the model more robust

#____________________________________________initialize the model
#select the device to use for training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#select the model to train (uncomment the model you want to use)
#model  = SimpleAutoencoder(IMG_SIZE,IMG_SIZE, latent_dim=LATENT_DIM, device=DEVICE)
#model = VGG16Autoencoder(IMG_SIZE,IMG_SIZE, latent_dim=LATENT_DIM, device=DEVICE)
model = VariationalAutoencoder(IMG_SIZE,IMG_SIZE, latent_dim=LATENT_DIM, device=DEVICE)

#move the model to the selected device
model.to(DEVICE)
#randomly initialize the weights of the model
model.initWeights()
#set the model to training mode (in pytorch this enables certain layers like dropout which are only used during training)
model.train()

#log the model architecture
print(model) #this prints the model layers and the number of parameters in each layer as seen by poytorch
model.printShapes((IMG_SIZE, IMG_SIZE), DEVICE) #this prints the shapes of the tensors at each layer

#initialize the dataloader
#set the transformation to apply to the images as they are loaded before showing them to the model
if RANDOMIZE:
    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE*1.05)), 
        transforms.RandomCrop(IMG_SIZE),  
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.1*torch.randn_like(x)),
        ])
else:
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE), 
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        ])

#create the datset and dataloader
dataset = ImageDataset(DATA_FOLDER, transform=transform)
#the batch size is the number of images to show to the model at once
#larger batch sizes can speed up training but require more memory
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

#randomly select 4 image indices to show during training so that we can monitor the progress
images_to_show = random.sample(range(len(dataset)), 4)

#initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

#train the model
for epoch in range(EPOCHS):
    total_loss = 0 #initialize the total loss for this epoch, we'll add up the loss for each image and then divide by the number of images to get the average loss

    for img in dataloader: #iterate through the batches of images
        #img here is a tensor with shape (batch_size, channels, height, width)
        #________________________copy image data to the selected device
        img = img.to(DEVICE)       
        #________________________reset the optimizer
        optimizer.zero_grad()         
        #________________________forward (pass the image through the model) encode(img)->latent_vector->decode(latent_vector)->output_image
        output = model(img)
        #________________________compute loss (compare the output to the original image), usually this is the mean squared error between the output and the original image's pixels
        loss= model.computeLoss(output, img)
        #________________________backward (compute the gradients)
        loss.backward()
        #________________________optimize (update the weights of the model)
        optimizer.step()

        #add the loss for this image to the total loss for this epoch
        total_loss += loss.sum().item()

    #compute the average loss for this epoch
    total_loss /= len(dataset)

    #print the average loss for this epoch
    print(f"Epoch {epoch+1}, Loss: {total_loss}")
    if (epoch+1) % SAVE_EVERY == 0:
        #save the model every SAVE_EVERY epochs
        model.save(MODEL_FOLDER)

    #show the images to monitor the progress
    #this is using opencv to open a window and show a grid of pairs of (input/output) images
    with torch.no_grad():
        img_pairs = []
        for img in images_to_show:
            img = dataset[img].unsqueeze(0).to(DEVICE)
            output = model(img)
            output = tensor_to_opencv(output)
            img = tensor_to_opencv(img)
            img_pairs.append(np.hstack((img, output)))
        img_stack = np.vstack(img_pairs)
        cv2.namedWindow('images', cv2.WINDOW_NORMAL)        
        cv2.resizeWindow('images', 2*128, len(img_pairs)*128)
        cv2.imshow("images", img_stack)
        cv2.waitKey(1)


#save the final model
model.save(MODEL_FOLDER)
