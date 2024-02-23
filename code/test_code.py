# %%
import tensorflow as tf
#import matplotlib.pyplot as plt  # For visualization (optional)

# %%
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

# %%
from tensorflow.keras.layers import Concatenate


# %%
def extract_images_complete(path):
  # count how many images inside our file (at this level we are considering the single folder video)
  import os

  previous = []
  following = []
  for i in path:
    dir_path_pre = '/home/shared-nearmrs/endovis_mono/' + i 
    dir_path = '/home/shared-nearmrs/endovis_mono/' + i + '/frame_data'
    count = 0
    # Iterate directory
    for pat in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, pat)):
            count += 1
    print('File count:', count)

    file_num = 000000
    a =[0]*count
    for i in range(0,count):
      image_path = dir_path_pre + '/frame_rgb/'+ str(file_num + i).zfill(6)+'.png'
      # Load the image without resizing or preprocessing
      image_1 = tf.io.read_file(image_path)
      image_1= tf.image.decode_image(image_1, channels=3)
      a[i] = image_1 #it's a list of images

    start_images = a[:count-1]
    final_images = a[1:]

    previous = previous + start_images
    following = following + final_images


  image_dataset_previous = np.array(previous) #with this command w create an array
  image_dataset_following = np.array(following) #with this command w create an array

  return image_dataset_previous,image_dataset_following




# %%
[x_pre,x_post] = extract_images_complete(['v1'])

x_pre = tf.image.resize(x_pre, [512,640])
x_post = tf.image.resize(x_post, [512,640])

print(x_pre.shape)


##########
import json

# %%
def matrix_to_rpy(R): # i think that this type of representation is not the correct one, since we have some ambiguity (the quaternians are better)
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return [roll, pitch, yaw]

# %%
def tranformation_matrix_extrac_2(folder_path):
  # looking for number of files in the folder
  final_position = []
  final_orientation = []
  for k in folder_path:
    import os
    dir_path = '/home/shared-nearmrs/endovis_mono/' + k + '/frame_data'
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
      # check if current path is a file
      if os.path.isfile(os.path.join(dir_path, path)):
          count += 1
  
    file_num = 000000
    tranf_matrices = [0]*(count-1)
    for i in range(0,count-1): #the upper boundar is exclude
      j = i + 1
      source = dir_path + '/frame_data' + str(file_num + i).zfill(6) + '.json'
      end = dir_path + '/frame_data' + str(file_num + j).zfill(6) + '.json'
  
      f_source = open(source)
      data_source = json.load(f_source)
      T_0 = np.array(data_source['camera-pose'])
  
      f_end = open(end)
      data_end = json.load(f_end)
      T_1 = np.array(data_end['camera-pose'])
  
      tranf_matrices[i] = np.matmul(np.linalg.inv(T_0),T_1)
    
    targets = np.stack(tranf_matrices, axis=0)

    position = [0]*targets.shape[0]
    angles = [0]*targets.shape[0]
    for r in range(0,targets.shape[0]):
      position[r] = targets[r,0:3,3]
      angles[r] = matrix_to_rpy(targets[r,:,:])
    
    #pos_aum = [i * 10 for i in position]
    pos_aum = position
    final_position = final_position + pos_aum
    final_orientation = final_orientation + angles
  
  a = np.stack(final_position, axis=0)
  b = np.stack(final_orientation, axis=0)


  return a,b


# MY_LOSS_FUNCTION 
from keras import backend as K
def my_loss_function(alpha_pos,alpha_orient):
    def loss(y_true, y_pred): #called each iteration (step)

        pred_pos, pred_orient = y_pred[:, :3], y_pred[:, 3:]
        target_pos, target_orient = y_true[:, :3], y_true[:, 3:]

        loss_pos = K.sum(K.abs(pred_pos - target_pos))
        loss_orient = K.sum(K.abs(pred_orient - target_orient))

        total_loss = alpha_pos * loss_pos + alpha_orient * loss_orient
       
        return total_loss
    
    return loss 




# %%
[a,b] = tranformation_matrix_extrac_2(['v1'])
y_pos = pd.DataFrame(a)
y_ori = pd.DataFrame(b)

# %%
y = pd.concat([y_pos,y_ori],axis = 1)
labels = ['px','py','pz','alpha','beta','gamma']
y.columns = labels
print(y)

x1 = np.array(x_pre/255)
x2 = np.array(x_post/255)
y = np.array(y)

# model predictionsù
# INDERST MODEL TO USE <--------------------------------------------------------------------------------------------------
# INDERST MODEL TO USE <--------------------------------------------------------------------------------------------------
a = 'my_model_final_albe_7'
model = tfk.models.load_model('/home/anaclerio/'+a+'/model')
#model = tfk.models.load_model('/home/anaclerio/'+a+'/model')
predictions = model.predict([x1,x2], verbose=0)
print(predictions.shape)
predictions = predictions/10

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate some random 3D points for demonstration
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], c='b', marker='o')

def trajectory(b):
  a = [0]*(b.shape[0]+1)
  a[0] = [0, 0, 0]
  for ii in range(0,b.shape[0]):
    a[ii+1] = a[ii]+b[ii,:3] #i want only the position
  a = np.array(a)
  return a

#print(np.array(Y_global).shape)
Y_global = trajectory(y)
Y_pred = trajectory(predictions)
np.save('/home/anaclerio/'+a+'/y_global.npy', Y_global)
np.save('/home/anaclerio/'+a+'/y_pred.npy', Y_pred)

np.save('/home/anaclerio/'+a+'/target_full.npy', y)
np.save('/home/anaclerio/'+a+'/predictions_full.npy', predictions)


def plotting(dimension):
  if dimension == 2:
    ax.scatter(Y_global[:, 0], Y_global[:, 1], c='r', marker='x')
    ax.scatter(Y_pred[:, 0], Y_pred[:, 1], c='b', marker='x')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('2D Scatter Plot')
    plt.savefig('/home/anaclerio/'+ a +'/2d_predictions_true.png')    
    plt.show()

    import plotly.graph_objs as go
    trace1 = go.Scatter3d(
        x=Y_global[:, 0],
        y=Y_global[:, 1],
        mode='markers',
        marker=dict(size=1, color='green')
    )

    trace2 = go.Scatter3d(
        x=Y_pred[:, 0],
        y=Y_pred[:, 1],
        mode='markers',
        marker=dict(size=1, color='red')
    )
    layout = go.Layout()
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.write_html('/home/anaclerio/' + a + '/interactive_plot_2.html')


  elif dimension == 3:
    ax.scatter(Y_global[:, 0], Y_global[:, 1], Y_global[:, 2], c='r', marker='x')
    ax.scatter(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], c='b', marker='x')
    ax.set_title('3D Scatter Plot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.savefig('/home/anaclerio/'+ a +'/3d_predictions_true.png')
    plt.show()
     
    import plotly.graph_objs as go

    # Create a 3D scatter plot
    trace1 = go.Scatter3d(
        x=Y_global[:, 0],
        y=Y_global[:, 1],
        z=Y_global[:, 2],
        mode='markers',
        marker=dict(size=3, color='green')
    )

    trace2 = go.Scatter3d(
        x=Y_pred[:, 0],
        y=Y_pred[:, 1],
        z=Y_pred[:, 2],
        mode='markers',
        marker=dict(size=3, color='red')
    )

    layout = go.Layout(scene=dict(aspectmode="cube"))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(scene=dict(aspectmode="cube"))
    fig.write_html('/home/anaclerio/'+ a +'/interactive_plot_3.html')

   
plotting(3)


#[5:46 PM] Ali Shadman Yazdi
#import numpy as np
# 
#def euler_angles_to_rotation_matrix(angles):
#    phi, theta, psi = angles
#    R_x = np.array([[1, 0, 0],
#                    [0, np.cos(phi), -np.sin(phi)],
#                    [0, np.sin(phi), np.cos(phi)]])
#    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
#                    [0, 1, 0],
#                    [-np.sin(theta), 0, np.cos(theta)]])
#    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
#                    [np.sin(psi), np.cos(psi), 0],
#                    [0, 0, 1]])
#    R = np.dot(R_z, np.dot(R_y, R_x))
#    return R
# 
#def angular_distance(angles1, angles2):
#    R1 = euler_angles_to_rotation_matrix(angles1)
#    R2 = euler_angles_to_rotation_matrix(angles2)
#   
#    trace = np.trace(np.dot(np.linalg.inv(R1), R2))
#    trace = min(3.0, max(-1.0, trace))  # Ensure trace is within valid range [-1, 3]
#   
#    angle_distance = np.arccos((trace - 1.0) / 2.0)
#    return np.degrees(angle_distance)
# 
#def mean_angular_error(predicted_angles, ground_truth_angles):
#    errors = [angular_distance(pred, gt) for pred, gt in zip(predicted_angles, ground_truth_angles)]
#    return np.mean(errors)
# 
## Example usage:
#predicted_angles = predictions[:,3:]
#ground_truth_angles = y_test[:,3:]
# 
#mae = mean_angular_error(predicted_angles, ground_truth_angles)
#print(f'Mean Angular Error: {mae} degrees')
# 
#




