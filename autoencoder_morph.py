#pip install pytorch_msssim
#pip install opencv-python
#pip install scikit-learn

import torch
from autoencoder import *
import os
import cv2

#__________________________________________________________________________________
#this script will load two images, encode them using the trained autoencoder
#and then interpolate between the two image vectors to create a morphing effect
#each interpolated vector is decoded into an image and saved to disk
#__________________________________________________________________________________

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#folder to save the morphed images
OUTPUT_FOLDER = os.path.join(THIS_FOLDER , 'images/variational_modigliani_128_morphed/')

#folder with the trained model
MODEL_FOLDER = "/Users/mac/Desktop/School/Harvard/08_autoencoder_scripts/model"

#images to morph
IMAGE_A = r'data/modigliani_128/amedeo-modigliani_a-woman-with-velvet-ribbon.jpg'
IMAGE_B = r'data/modigliani_128/amedeo-modigliani_young-woman-1910.jpg'

#number of steps in the morphing process
STEPS = 10

#load torch traced model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder(MODEL_FOLDER, DEVICE)

print("IMG_SIZE " , autoencoder.image_width, autoencoder.image_height)
print("LATENT_DIM", autoencoder.latent_dim)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#encode the two images to get their latent space vectors
image_a_vec = autoencoder.encode(IMAGE_A)
image_b_vec = autoencoder.encode(IMAGE_B)


#interpolate between the two images
for i in range(STEPS):
    #t is the interpolation factor. t=0 is image A, t=1 is image B, t=0.5 is the average of A and B
    #t increases from 0.0 to 1.0 as the loop progresses from 0 to STEPS-1
    t = i/(STEPS-1)
    
    #find the interpolated vector
    image_vec = image_a_vec + (image_b_vec-image_a_vec)*t

    #decode the interpolated vector into an image tensor
    img = autoencoder.decode(image_vec)

    #convert the tensor image to opencv format and save it
    img = tensor_to_opencv(img)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'{i}.jpg'), img)


