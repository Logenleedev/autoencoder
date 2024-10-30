import os
from PIL import Image

image_size = 128
skip = 1

srcFolder = r"D:\ML\CLASS_DEMOS\autoencoder_models\modigliani_temp"
dstFolder = r'D:\Dropbox\ACADEMIC\GSD\2024_02_Fall_QA\08_autoencoders\scripts\data\modigliani_128'

os.makedirs(dstFolder, exist_ok=True)

image_ext = ['jpg', 'jpeg', 'png']
files = os.listdir(srcFolder)
files = [f for f in files if f.split('.')[-1].lower() in image_ext]

for i in range(0, len(files), skip):
    filename = files[i]
    img = Image.open(os.path.join(srcFolder, filename))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    #crop center
    width, height = img.size
    new_width = min(width, height)
    new_height = new_width
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img = img.crop((left, top, right, bottom))
    img = img.resize((image_size,image_size))
    img.save(os.path.join(dstFolder, filename))