#pip install pytorch_msssim
#pip install opencv-python
#pip install scikit-learn

import torch
import os
import numpy as np
from autoencoder import *
import cv2

#__________________________________________________________________________________
#this script will load a set of images, encode them into latent vectors using the trained autoencoder
#and then recode those vectors back into images using the decoder part of the autoencoder
#this is a test to verify that the autoencoder can encode and decode images correctly
#it will also save the latent vectors to a csv file for further analysis
#__________________________________________________________________________________

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

#data folder with images to use
DATA_FOLDER = os.path.join(THIS_FOLDER, 'data/modigliani_128')

#folder to save the recoded images
OUTPUT_FOLDER = os.path.join(THIS_FOLDER , 'images/modigliani_128_recoded/')

#folder with the trained model
MODEL_FOLDER = "/Users/mac/Desktop/School/Harvard/08_autoencoder_scripts/model"

#load torch traced model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder(MODEL_FOLDER, DEVICE)


print("IMG_SIZE " , autoencoder.image_width, autoencoder.image_height)
print("LATENT_DIM", autoencoder.latent_dim)


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

extensions = ['jpg', 'jpeg', 'png']
images = [f for f in os.listdir(DATA_FOLDER) if f.split('.')[-1].lower() in extensions]


#load all images
#create an empty list to store the latent vectors for each image
img_vectors = []
for img_name in images:
    #construct the full path to the image file
    img_path = os.path.join(DATA_FOLDER, img_name)
    #encode the image to get the latent vector
    img_vector = autoencoder.encode(img_path)
    #decode the latent vector to get the recoded image
    recoded = autoencoder.decode(img_vector)
    #append the latent vector to the list
    img_vectors.append(img_vector.cpu().numpy().flatten())

    #convert the tensor image to opencv format and save it
    recoded = tensor_to_opencv(recoded)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, img_name), recoded)


#save csv file
#open a csv file for writing to save the latent vectors
with open(os.path.join(OUTPUT_FOLDER, 'img_vectors.csv'), 'w') as f:
    #loop over the images and their indices so that we can grab the correspong latent vector img_vector[i] for each image
    for i, image_file in enumerate(images):
        #write the image file name at the start of the csv file's row
        f.write(f"{image_file}")
        #write the latent vector values for the image
        for j in img_vectors[i]:
            #write each component of the latent vector as a number with 4 decimal places preceded by a comma
            f.write(f",{j : .4f}")
        #write a newline character to move to the next row
        f.write("\n")

