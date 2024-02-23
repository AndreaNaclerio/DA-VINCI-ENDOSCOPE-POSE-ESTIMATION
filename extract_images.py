import tensorflow as tf
import numpy as np
#import os
#import random
#import pandas as pd
#import seaborn as sns
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
#from sklearn.metrics import confusion_matrix
#import tensorflow as tf
#import matplotlib.pyplot as plt  # For visualization (optional)
#tfk = tf.keras
#tfkl = tf.keras.layers
#print(tf.__version__)

import sys

def extract_images(path):
  # count how many images inside our file (at this level we are considering the single folder video)
  import os
  dir_path = path +'/frame_data'
  count = 0
  # Iterate directory
  for pat in os.listdir(dir_path):
      # check if current path is a file
      if os.path.isfile(os.path.join(dir_path, pat)):
          count += 1
  print('File count:', count)

  count = 197
  file_num = 000000
  a =[0]*count 
  for i in range(0,count):
    image_path = path + '/frame_rgb/'+ str(file_num + i).zfill(6)+'.png'
    # Load the image without resizing or preprocessing
    image_1 = tf.io.read_file(image_path)
    image_1= tf.image.decode_image(image_1, channels=3)
    a[i] = image_1 #it's a list of images

  image_dataset = np.stack(a, axis=0) #with this command w create an array
  print(image_dataset.shape)
  return image_dataset

extract_images(sys.argv[1])
