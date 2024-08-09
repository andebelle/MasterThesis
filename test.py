# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:08:48 2024

@author: Anouk
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import sys
from skimage import exposure
from skimage.exposure import match_histograms 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_curve, auc
import os
import matplotlib.pyplot as plt
import seaborn as sns
path_utils = "C:/Documents/Université/Master/Mémoire/Code/example/utils"
if(not(path_utils in sys.path)):
    sys.path.insert(0,path_utils)
input_path = 'C:/Documents/Université/Master/Mémoire/Code/example/Real_set'
import mnist_utils as mnu

#with open('C:/Documents/Université/Master/Mémoire/Code/example/MNIST/new_x_test_withoutnoise_all_skeletonLee_3.pickle', 'rb') as f:
with open('C:/Documents/Université/Master/Mémoire/Code/example/Real_set/x_val.pickle', 'rb') as f:
    x = pickle.load(f)
    
#with open('C:/Documents/Université/Master/Mémoire/Code/example/MNIST/new_y_test_withoutnoise_all_skeletonLee_3.pickle', 'rb') as f:
with open('C:/Documents/Université/Master/Mémoire/Code/example/Real_set/y_val.pickle', 'rb') as f:
    y = pickle.load(f)

    
  

x = mnu.min_max(x, axis = (1,2) )
x = 1 - x 

#%%
idx = np.random.choice(len(x),size=1,replace=False)

plt.imshow(x[idx].squeeze(), cmap='turbo')
plt.colorbar()
plt.title('Sample : {0} | Label : {1}'.format( idx, str(y[idx])))
plt.show()



#%%

#sample = np.random.choice(len(x),size=1,replace=False)

hist, bin_edges = np.histogram(x,bins=20, density=False)

bin_middle = (bin_edges[1:]+bin_edges[:-1])/2

fig = plt.figure()
ax = fig.add_subplot(1,2,1)

ax.plot(bin_middle,hist,'oc')
#ax.set_title('Sample : {0} | Label : {1}'.format( sample, str(y[sample])))
"""
ax = fig.add_subplot(1,2,2)

mp = ax.imshow(x[sample,...].squeeze(),cmap='gray')
ax.set_title('Sample : {0} | Label : {1}'.format( sample, str(y[sample])))
plt.colorbar(mp)
plt.show()

#np.sum(x[sample]>0.99)
"""
#%%
plt.figure()
plt.imshow(x[5,...],cmap='turbo')
plt.colorbar()
plt.show()


#%%
x = torch.from_numpy(x[:,np.newaxis,...]).float()
y = torch.from_numpy(y.astype('int64'))

#%%
model = torch.load('C:/Documents/Université/Master/Mémoire/Code/example/models/model_withoutnoise_all_skeletonLee_3.pt', map_location=torch.device('cpu'))
#model.eval()
with torch.no_grad():
    o = mnu.evaluate(model, x, (x.shape[0],13)).detach().cpu()
pred = torch.exp(o)/torch.sum(torch.exp(o),dim=-1,keepdims=True)
digits_pred_test = torch.argmax(pred, dim = -1)

#%%
sample = np.random.randint(x.shape[0])
#sample=0

plt.figure()
plt.imshow(x[sample,...].squeeze(), cmap = 'turbo', )
plt.colorbar()
plt.title('Sample : {0} | GT label : {1} | Predicted label : {2} '.format(sample, 
                                                                          y[sample], 
                                                                          digits_pred_test[sample],))
plt.show()

#%%

acc_test = torch.sum(y == digits_pred_test)/y.shape[0]
print(' Test accuracy : ', acc_test)

#%%

precision_test = precision_score(y.numpy(), digits_pred_test.numpy(), average='macro')
recall_test = recall_score(y.numpy(), digits_pred_test.numpy(), average='macro')

f1_test = f1_score(y.numpy(), digits_pred_test.numpy(),average='macro')

print('Precision: ', precision_test)
print('Recall: ', recall_test)
print('F1 score: ', f1_test)

#%%
classes = torch.unique(y)
class_correct = torch.zeros(13)
class_total = torch.zeros(13)

# Compter les prédictions correctes pour chaque classe
for i in range(len(y)):
    label = y[i]
    pred = digits_pred_test[i]
    class_total[label] += 1
    if label == pred:
        class_correct[label] += 1
# Calculer la précision pour chaque classe
class_accuracy = class_correct / class_total

# Afficher la précision pour chaque classe
for i in range(13):
    print('Test accuracy for the class : {0} : {1}'.format(i,class_accuracy[i]))
    
#%%
conf_matrix = confusion_matrix(y.numpy(), digits_pred_test.numpy())

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediction')
    plt.ylabel('Real label')
    plt.title('Confusion matrix for the real dataset')
    plt.show()

# Noms des classes (ajustez en fonction de votre problème)
class_names = ['0','1','2','3','4','5','6','7','8','9','10','11','12']

# Affichage de la matrice de confusion
plot_confusion_matrix(conf_matrix, class_names)
