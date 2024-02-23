import numpy as np
a = 'my_model_final_albe_7'

y_true= np.load('/home/anaclerio/' + a + '/y_global.npy')
y_pred= np.load('/home/anaclerio/' + a + '/y_pred.npy')
diff = abs(y_true-y_pred)
print(diff)

#def trajectory(b):
#  a = [0]*(b.shape[0]+1)
#  a[0] = [0,0,0]
#  for ii in range(0,b.shape[0]):
#    a[ii+1] = a[ii]+b[ii,:3] #i want only the position
#  a = np.array(a)
#  return a
#Y_global = trajectory(y_true)
#Y_pred = trajectory(y_pred)

import math
def euclidean_distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

distances=[]
for i in range(len(y_true)):
  distances.append(euclidean_distance_3d(y_true[i],y_pred[i]))

distances = np.array(distances)
print(distances)

#distances = [0]*Y_global.shape[0]
#for i in range(0,Y_global.shape[0]):
#  distances[i] =math.sqrt((Y_global[i][0] - Y_pred[i][0])**2 + (Y_global[i][1] - Y_pred[i][1])**2 + (Y_global[i][2] - Y_pred[i][2])**2)
#
#print(distances)

x  = np.arange(0,y_true.shape[0],1)
#print(x)
#
import numpy as np
import matplotlib.pyplot as plt
#axs.set_title("Patient:" + str(patient) + " - Label:" + labels_adapt[patient])
plt.plot(x,  distances, color='C0',label='X')
plt.legend('Euclidian distances between predicted-true points')
plt.xlabel("points")
plt.ylabel("distance")
plt.savefig('/home/anaclerio/' + a + '/distances.png')

plt.show()



