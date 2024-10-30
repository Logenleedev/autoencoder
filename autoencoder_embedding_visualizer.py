#pip install pytorch_msssim
#pip install opencv-python
#pip install scikit-learn

import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from autoencoder import *

#this is needed to compute the 2d projection of the images' latent space vectors
from sklearn.manifold import TSNE

#__________________________________________________________________________________
#this script will load a set of images, encode them using the trained autoencoder
#and then project the encoded vectors into 2D space using t-SNE
#matplotlib will be used to display the images as thumbnails projected in 2D space
#__________________________________________________________________________________

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#data folder with images to use for display
DATA_FOLDER = os.path.join(THIS_FOLDER, 'data/modigliani_128')
#folder with the trained model
MODEL_FOLDER = "/Users/mac/Desktop/School/Harvard/08_autoencoder_scripts/model"

#parameters for the display of the images' thumbnails
THUMBNAIL_OPACITY = 0.8
BLUR_THUMBNAIL_EDGES = False    #set to True to blur the edges of the thumbnails
THUMBNAIL_SIZE = 0.025          #size of the thumbnails in fraction of the plot size

#load torch traced model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Autoencoder is defined in autoencoder.py. It helps to load the model and encode/decode images
autoencoder = Autoencoder(MODEL_FOLDER, DEVICE)


print("IMG_SIZE " , autoencoder.image_width, autoencoder.image_height)
print("LATENT_DIM", autoencoder.latent_dim)

#collect all valid image files in the data folder
extensions = ['jpg', 'jpeg', 'png']
images = [f for f in os.listdir(DATA_FOLDER) if f.split('.')[-1].lower() in extensions]


#load each image and encode it
img_vectors = []
for img_name in images:
    img_path = os.path.join(DATA_FOLDER, img_name)
    img_vector = autoencoder.encode(img_path)
    img_vectors.append(img_vector.cpu().numpy().flatten())

#concatenate all the image vectors into a single numpy array, a matrix where each row is an image vector
img_vectors = np.array(img_vectors)

#reduce dimensionality using t-SNE from sklearn to project the image vectors into 2D space
#tsne will try to preserve the local structure of the data by minimizing the distance between similar images
tsne = TSNE(n_components=2, perplexity=20.0)
img_points_2d = tsne.fit_transform(img_vectors)


#compute the x and y range of the 2D points to set the plot limits
min_x = np.min(img_points_2d[:, 0])
max_x = np.max(img_points_2d[:, 0])
min_y = np.min(img_points_2d[:, 1])
max_y = np.max(img_points_2d[:, 1])

range_x = max_x - min_x
range_y = max_y - min_y

#compute the size of the thumbnails in the plot relative to the plot size
thumbnail_size = min(range_x, range_y)*THUMBNAIL_SIZE

#plot using matplotlib
#initialize the plot
plt.figure(figsize=(10, 10))
#set the plot limits
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)

#load each and position the image thumbnails in the plot
for i, image_file in enumerate(images):
    #load the image
    img = cv2.imread(os.path.join(DATA_FOLDER, image_file), cv2.IMREAD_COLOR)
    #convert the image to RGBA format (openCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
   
    #set the alpha channel to the thumbnail opacity
    if BLUR_THUMBNAIL_EDGES:
        img_x = np.linspace(-1.0, 1.0, img.shape[1])
        img_y = np.linspace(-1.0, 1.0, img.shape[0])

        grid = np.meshgrid(img_x, img_y)
        img_x = grid[0]
        img_y = grid[1]
        alpha = (np.exp(-4.0*(img_x**2 + img_y**2))*2.0).clip(0.0, 1.0)*255.0*THUMBNAIL_OPACITY
        img[:,:,3] = alpha[:]
    else:
        img[:,:,3] = THUMBNAIL_OPACITY*255.0

    #read the x and y ([0], [1]) coordinates of the 2D projection 
    x = img_points_2d[i, 0]
    y = img_points_2d[i, 1]
    #plot the image as a thumbnail centered at the x, y coordinates
    plt.imshow(img, extent=(x-thumbnail_size, x+thumbnail_size, y-thumbnail_size, y+thumbnail_size), aspect='auto')

#show the plot
plt.show()

