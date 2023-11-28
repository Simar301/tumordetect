import cv2
import os
from PIL import Image
from keras.models import load_model
import numpy as np
model=load_model('simar.h5')
image_directory='datasets/'
pred_tumor_images=os.listdir(image_directory+ 'pred/')
for i , image_name in enumerate(pred_tumor_images):#looping in the folder 
    if(image_name.split('.')[1]=='jpg'):#selecting all the file with extension jpg
        image=cv2.imread(image_directory+'pred/'+image_name)#reading image from the file
        image=Image.fromarray(image,'RGB')#converting image from(BGR)format to RGB format
        image=image.resize((64,64))#converting image to 64*64 for consistency
        img=np.array(image)
        input_img=np.expand_dims(img,axis=0)
        result=model.predict(input_img)
        if result==0:
               print("No brain tumor")
        else:
              print("brain tumor")




