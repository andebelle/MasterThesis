# -*- coding: utf-8 -*-
import importlib
import numpy as np # linear algebra
import struct
from array import array
import matplotlib.pyplot as plt

torch = importlib.import_module('torch')
modules_dic = {}
modules_dic['torch'] = torch

from tqdm import tqdm
import time
from functools import reduce

#%%
def SolveAttr(obj, attr):
    try:
        a = reduce(getattr, attr, obj)
    except AttributeError:
        raise AttributeError(f'Couldnt solve the attribute {attr} for object {obj}')
    return a

#%%
def min_max(inputs, axis = (2,3)):
    minis = np.min(inputs, axis = axis, keepdims = True)
    maxis = np.max(inputs, axis = axis, keepdims = True)
    return (inputs - minis)/(maxis - minis)

#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%%
def create_new_digits(x,y, first_digit, second_digit, num_new_digits, 
                      first_start = 0, second_start = 20, 
                      first_inter_length = 12, second_inter_length = 8):
    
    (num_samples, ax1_size, ax2_size) = x.shape
    new_ax2_size = second_start + second_inter_length + ax2_size
    new_x = np.zeros((num_new_digits, ax2_size,new_ax2_size), dtype = 'float32')
    dumb_mask = np.ones((num_new_digits, *x.shape[1:]), dtype = 'bool')
        
    mask_1 = y == first_digit  
    if(np.sum(mask_1)!=0):
        first_digits_idx = np.random.choice(np.sum(mask_1), num_new_digits)    
    
        first_indexes = np.random.randint(first_start,first_start+first_inter_length, 
                                          size = (num_new_digits,1)) + np.array([[k for k in range(ax2_size)]])
        first_indexes = np.tile(first_indexes[:,np.newaxis,:], reps = (1,28,1))
        mask_first_pos = np.zeros(new_x.shape, dtype = 'bool')
        np.put_along_axis(mask_first_pos, first_indexes, 1, axis = -1)
        
        new_x[mask_first_pos] = x[mask_1,...][first_digits_idx,...][dumb_mask]
    
    mask_2 = y == second_digit
    if(np.sum(mask_2)!=0):
        second_digits_idx = np.random.choice(np.sum(mask_2), num_new_digits)
        second_indexes = np.random.randint(second_start,second_start+second_inter_length, 
                                           size = (num_new_digits,1)) + np.array([[k for k in range(ax2_size)]])
        second_indexes = np.tile(second_indexes[:,np.newaxis,:], reps = (1,ax1_size,1))
        mask_second_pos = np.zeros(new_x.shape, dtype = 'bool')
        np.put_along_axis(mask_second_pos, second_indexes, 1, axis = -1)
    
        
        new_x[mask_second_pos] = new_x[mask_second_pos] + x[mask_2,...][second_digits_idx,...][dumb_mask]
    
        new_x[new_x>1] = 1
    return new_x


#%%
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  
    
    
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
        
    
#%%
class Network(torch.nn.Module):
    def __init__(self, architecture, skips = None):
        super(Network, self).__init__()
        self.architecture = architecture
        self.Layers = torch.nn.Sequential()
        self.skips = skips
        if(not(skips is None)):
            self.skips = skips
        for i,module_data in enumerate(architecture):
            full_name = module_data[0].split('.')
            name = full_name[1:]
            args = module_data[1]
            self.Layers.add_module('Layer_{0}'.format(i), SolveAttr(modules_dic[full_name[0]], name)(**args))
                    
            
    def reset_parameters(self):
        for layer in self.Layers:
            try:
                layer.reset_parameters()
            except AttributeError:
                print('Could not initalize weights for : ', layer)
            
        return None
    
    def forward(self, x, verbosity = 0):
        idx_skip = 0
        for i,module in enumerate(self.Layers):
            if(verbosity>0):
                print('==== module : ', module)
            x = module(x)
            if(not(self.skips is None) and idx_skip < len(self.skips[0])):

                if(i==self.skips[0][idx_skip]):
                    if(verbosity>0):
                        print('idx skip : ', idx_skip)
                        print('i : ', i)
                    stored_x = torch.clone(x)
                elif(i==self.skips[1][idx_skip]):
                    if(verbosity > 0):
                        print('stored_x : ', stored_x.shape)
                        print('x : ', x.shape)
                        print('idx skip : ', idx_skip)
                        print('i : ', i)
                    x = x + stored_x
                    idx_skip+=1
                
            if(verbosity > 0):
                print('========== ')
                print(x.shape)
        return x
                    


#%%
def evaluate(model, x, out_shape, chunksize = None):
    if(chunksize is None):
        out = model.forward(x)
    else:
        idx_start = 0
        num_chunks = x.shape[0]//chunksize
        num_chunks += int(x.shape[0] - (num_chunks * chunksize) > 0)
        out = torch.zeros(out_shape, dtype = x.dtype, device = x.device)
        for j in range(num_chunks):
            idx_end = idx_start + chunksize
            out[idx_start:idx_end,...] = model.forward(x[idx_start:idx_end,...])
            idx_start = idx_end
    return out
            

def update_log(msg, logfile):
    with open(logfile, 'a') as f:
        f.write(msg)
        
            

#%%
def Train(model,
          optimizer,
          batch_size,
          num_epochs, 
          in_train, in_valid, 
          target_train,target_valid, 
          train_chunk_size = None,
          valid_chunk_size = None,
          device = 'cuda',
          verbosity = 0,
          logfile = None,
          random_seed = 10, 
          shuffling = False,
          loss_function = None,
          metric_function = None, ):
        
        if(logfile is None):    
            update_log_func = print
        else:
            with open(logfile, 'w') as f:
                f.write('============ NEW LOGFILE \n\n')
            update_log_func = lambda msg:update_log(msg, logfile)
        if(verbosity > 10):
            max_grad = 0 #Maximum gradient values (over all parameters) : to monitor training
        torch.manual_seed(random_seed)
        num_train_samples = in_train.shape[0] #Number of samples in the training set
        
        if(not(shuffling)):
            get_slice = lambda i, size: range(i * size, (i + 1) * size) #Will be used to get the different batches
        
        
        if(loss_function is None):
            update_log_func('\nWARNING : Using default (L1Loss) loss function')
            loss_function = torch.nn.L1Loss(reduction = 'mean')
            
        #For 'accuracy' evaluation
        #Should be different than loss_fn ideally, 
        #and can be used to quickly determine the comparative performances of different loss functions
        if(metric_function is None):
            metric_function = torch.nn.L1Loss(reduction='mean') 
            
        
        update_log_func('\n----------------------- Training --------------------------\n\n')
        start = time.time() 
        
        num_batches_train = num_train_samples // batch_size #Number of training batches
        train_losses = [] #Will store the loss on the training set for each epoch
        val_losses = [] #Will store the loss on the validation set for each epoch
        val_acc = [] #Will store the 'accuracy' on the validation set for each epoch
        
        if(device == 'cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in tqdm(range(num_epochs)):
            if(verbosity > 1):
                update_log_func('\n************ Epoch {0} ************'.format(epoch))
            if(verbosity > 10):
                update_log_func('memory :', torch.cuda.memory_allocated()/1024/1024)
            for i in range(num_batches_train):
                optimizer.zero_grad() #Reset gradients
                batch = get_slice(i,batch_size) #Create indexes of current batch
                output = evaluate(model, in_train[batch,...], 
                                  (target_train[batch,...].shape[0],13), 
                                  chunksize = train_chunk_size)
                batch_loss = loss_function(output, target_train[batch, ...])
                batch_loss.backward() #Calculate gradients
                optimizer.step() #Update based on gradients
                
                if(verbosity >5):
                    full_grads = [x.grad for x in model.parameters()]
                    maxi = 0
                    min_grad = 1e12
                    for grad in full_grads:
                        if(not(grad is None)):
                            maxi = torch.max(torch.abs(grad))
                            mini = torch.min(torch.abs(grad))
                            max_grad = max(max_grad,maxi)
                            min_grad = min(min_grad,mini)
                    update_log_func('Epoch : {0} / Batch : {1} / Max grad so far : {2}'.format(epoch,i,max_grad))
                    update_log_func('Epoch : {0} / Batch : {1} / Min grad so far : {2}'.format(epoch,i,min_grad))
                if(verbosity>0):
                    update_log_func('\n=== Train loss : ' + str(batch_loss)) 
                    
            #Evaluate on validation set
            with torch.no_grad():
                validation_output = evaluate(model, in_valid, 
                                  (target_valid.shape[0],13), 
                                  chunksize = valid_chunk_size)
                loss_valid = loss_function(validation_output, target_valid.to(device),)
                val_losses.append(loss_valid.detach().cpu())
            if(verbosity>0):
                update_log_func('\nValidation loss at end of the epoch : ' + str(loss_valid))

        
        end = time.time()
        total_time = end-start
        out = {"train_losses": train_losses,
               "validation_losses": val_losses,
               "validation_accuracy" : val_acc,
               "model": model,
               "total_time": total_time,
               "optimizer": optimizer,
               }
        if(device == 'cuda'):
            torch.cuda.empty_cache()
        return out