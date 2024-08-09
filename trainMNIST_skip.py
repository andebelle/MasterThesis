# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt
import torch
import sys
path_utils = "/auto/home/users/a/d/adebelle/thesis/example/utils/"
if(not(path_utils in sys.path)):
    sys.path.insert(0,path_utils)
input_path = "/auto/home/users/a/d/adebelle/thesis/example/MNIST/"
import mnist_utils as mnu
import torch
import os
import json
import pickle

if __name__ == "__main__":
    #%%
    param = {'in_channels':[1,32,32,64,64,128,128], 'out_channels': [32,32,64,64,128,128,256], 'p_conv' : 0.2,
            'p_linear': 0.3, 'in_features':[256*3,256,200], 'out_features':[256,200,13], 'skips': [[0,7],[4,11]]}
    #%% Load
    with open(os.path.join(input_path, 'new_x_train_withoutnoise_lines.pickle'), 'rb') as f:
        x = pickle.load(f)
    
    with open(os.path.join(input_path, 'new_y_train_withoutnoise_lines.pickle'), 'rb') as f:
        y = pickle.load(f)
    
    with open(os.path.join(input_path, 'new_x_test_withoutnoise_lines.pickle'), 'rb') as f:
        x_test = pickle.load(f)
    
    with open(os.path.join(input_path, 'new_y_test_withoutnoise_lines.pickle'), 'rb') as f:
        y_test = pickle.load(f)

    shuffling_idx = np.random.choice(x.shape[0], size = x.shape[0], replace = False)
    num_train_samples = 180000
    x_train = x[shuffling_idx,...][0:num_train_samples,...][:,np.newaxis,...]
    x_valid = x[shuffling_idx,...][num_train_samples:,...][:,np.newaxis,...]

    y_train = y[shuffling_idx,...][0:num_train_samples,...]
    y_valid = y[shuffling_idx,...][num_train_samples:,...]

    #%%
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train.astype('int64'))
    x_valid = torch.from_numpy(x_valid).float()
    y_valid = torch.from_numpy(y_valid.astype('int64'))
    x_test = torch.from_numpy(x_test[:,np.newaxis,...]).float()
    y_test = torch.from_numpy(y_test.astype('int64')) 

    #%% 
    def architecture(in_channels, out_channels, p_conv, p_linear, in_features, out_features, skips):
    
      architecture = [ ('torch.nn.Conv2d', {'in_channels': in_channels[0], 'out_channels':out_channels[0], 'kernel_size':3,'padding':'same'}),
                  ('torch.nn.ReLU',{}),
                  ('torch.nn.Dropout2d',{ 'p': p_conv}),
                  ('torch.nn.Conv2d', {'in_channels': in_channels[1], 'out_channels':out_channels[1], 'kernel_size':3,'padding':'same'}),
                  ('torch.nn.ReLU', {}),
                  ('torch.nn.MaxPool2d',{ 'kernel_size':2}, {} ),
                  ('torch.nn.Dropout2d',{ 'p': p_conv}),
                  ('torch.nn.Conv2d', {'in_channels': in_channels[2], 'out_channels':out_channels[2], 'kernel_size':3,'padding':'same'}),
                  ('torch.nn.ReLU', {}),
                  ('torch.nn.Dropout2d',{ 'p': p_conv}),
                  ('torch.nn.Conv2d', {'in_channels': in_channels[3], 'out_channels':out_channels[3], 'kernel_size':3,'padding':'same'}),
                  ('torch.nn.ReLU', {}),
                  ('torch.nn.MaxPool2d',{ 'kernel_size':2} ),
                  ('torch.nn.Dropout2d',{ 'p': p_conv}),
                  ('torch.nn.Conv2d', {'in_channels': in_channels[4], 'out_channels':out_channels[4], 'kernel_size':3,'padding':'same'}),
                  ('torch.nn.ReLU', {}),
                  ('torch.nn.Dropout2d',{ 'p': p_conv}),
                  ('torch.nn.Conv2d', {'in_channels': in_channels[5], 'out_channels':out_channels[5], 'kernel_size':3,'padding':'same'}),
                  ('torch.nn.ReLU', {}),
                  ('torch.nn.MaxPool2d',{ 'kernel_size':2} ),
                  ('torch.nn.Dropout2d',{ 'p': p_conv}),
                  ('torch.nn.Conv2d', {'in_channels': in_channels[6], 'out_channels':out_channels[6], 'kernel_size':3,'padding':'same'}),
                  ('torch.nn.ReLU', {}),
                  ('torch.nn.MaxPool2d',{ 'kernel_size':2} ),
                  ('torch.nn.Dropout2d',{ 'p': p_conv}),
                  ('torch.nn.Flatten',{ 'start_dim':1, 'end_dim':-1}),
                  ('torch.nn.Dropout',{ 'p':p_conv} ),
                  ('torch.nn.Linear',{'in_features': in_features[0], 'out_features':out_features[0],}),
                  ('torch.nn.ReLU', {}),
                  ('torch.nn.Dropout2d',{ 'p': p_linear} ),
                  ('torch.nn.Linear',{'in_features': in_features[1], 'out_features':out_features[1],}),
                  ('torch.nn.ReLU', {}),
                  ('torch.nn.Dropout2d',{ 'p':p_linear} ),
                  ('torch.nn.Linear',{'in_features': in_features[2], 'out_features':out_features[2],}),
                ]
      skips = skips
      return architecture, skips

    arch, skips = architecture(param['in_channels'], param['out_channels'], param['p_conv'], param['p_linear'], param['in_features'], param['out_features'], param['skips'])
    #%%
    model = mnu.Network(arch, skips = skips).to('cuda')
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=7.5e-4)
    print('Total number of parameters : ', mnu.count_parameters(model))
    #%%
    out = model(x_train[0:50,...].to('cuda'), verbosity = 1)
    loss = torch.nn.CrossEntropyLoss(reduction = 'mean')
    l = loss(out, y_train[0:50].to('cuda'))
    #%% train
    logfile = "/globalscratch/users/a/d/adebelle/logs/log_skeletonLee_3.txt"
    resu = mnu.Train(model, optimizer, 10000, 30,
                     x_train.to('cuda'), x_valid.to('cuda'), 
                     y_train.to('cuda'), y_valid.to('cuda'), 
                     device = 'cuda', loss_function = loss, metric_function = loss,
                     verbosity = 2,
                     logfile = logfile,
                     train_chunk_size = 5000,
                     valid_chunk_size = 5000)

    #%%
    plt.figure()
    plt.plot(resu['train_losses'], '--k')
    plt.title('Loss curve of the training')
    plt.savefig(os.path.join("/globalscratch/users/a/d/adebelle/graphs/",'loss_graph_withoutnoise_lines.png'))
    plt.show()
    
    plt.figure()
    plt.plot(resu['validation_losses'], '-c')
    plt.title('Loss curve of the validation')
    plt.savefig(os.path.join("/globalscratch/users/a/d/adebelle/graphs/",'loss_graph_withoutnoise_lines.png'))
    plt.show()

    #%% some checks
    with torch.no_grad():
        temp = mnu.evaluate(model, x_test.to('cuda'), (x_test.shape[0], 13), chunksize = 1000).detach().cpu()

    #%%
    pred_test = torch.exp(temp)/torch.sum(torch.exp(temp), dim = -1, keepdims = True)
    digits_pred_test = torch.argmax(pred_test, dim = -1)


    #%% Sanity checks
    sample = np.random.randint(x_test.shape[0])

    plt.figure()
    plt.imshow(x_test[sample,...].squeeze(), cmap = 'binary_r', )
    plt.colorbar()
    plt.title('Sample : {0} | GT label : {1} | Predicted label : {2} '.format(sample, 
                                                                          y_test[sample], 
                                                                     digits_pred_test[sample],))
    plt.show()

    #%%
    num_test_samples = y_test.shape[0]
    samples = np.random.choice(num_test_samples, 10, replace = False)
    print('gt : ', y_test[samples])
    print('pred : ', digits_pred_test[samples])
    
    

    #%%
    acc_test = torch.sum(y_test == digits_pred_test)/num_test_samples
    print(' Test accuracy : ', acc_test) 


    #%% save model
    torch.save(model, os.path.join("/globalscratch/users/a/d/adebelle/models/", 'model_withoutnoise_lines.pt'))
    with open(os.path.join("/globalscratch/users/a/d/adebelle/Parameters_model/",'model_withoutnoise_lines.json'), 'w') as file:
      json.dump(param, file, indent=4)