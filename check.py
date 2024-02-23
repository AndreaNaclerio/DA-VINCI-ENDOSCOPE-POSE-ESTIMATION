#IMPORT LIBRARIES
import numpy as np
import tensorflow as tf
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)
from tensorflow.keras.layers import Concatenate

y = np.load('target.npy')
print(y.shape)
