# -*- coding: utf-8 -*-

# Model imports
import keras
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import Adam
from keras.optimizers import SGD
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from keras.layers import MaxPooling2D

# Tensorflow config
from keras import backend as K
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Other supplementary imports
import pickle
import numpy as np
import random
import h5py


##############################################################################
"""LOAD A PICKLE OBJECT"""
##############################################################################
def pickle_load(path):
    with open(path, 'rb') as handle:
        output = pickle.load(handle)
    return output


##############################################################################
"""SAVE A PICKLE OBJECT"""
##############################################################################
def pickle_save(variable_to_save, path):
    with open(path, 'wb') as handle:
        pickle.dump(variable_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

##############################################################################
"""DEFINE OPTIMIZER"""
##############################################################################
def define_optimizer(optimizer, learning_rate):
    if optimizer == 'adam':
        my_optimist = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer == 'adamax':
        my_optimist = Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif optimizer == 'nadam':
        my_optimist = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    elif optimizer == 'adadelta':
        my_optimist = Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
    elif optimizer == 'adagrad':
        my_optimist = Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
    elif optimizer == 'SGD':
        my_optimist = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        my_optimist = None
        print('Error on choosing optimizer. stopping program', error=True)
        exit()
    
    return my_optimist


##############################################################################
"""LOAD PCAM DATASET"""
##############################################################################
def load_PCAM():
    with h5py.File('pcam_data/camelyonpatch_level_2_split_train_x.h5', 'r') as f:
        trainX = f['x'][()]
    with h5py.File('pcam_data/camelyonpatch_level_2_split_train_y.h5', 'r') as f:
        trainY = f['y'][()]
    with h5py.File('pcam_data/camelyonpatch_level_2_split_valid_x.h5', 'r') as f:
        valX = f['x'][()]
    with h5py.File('pcam_data/camelyonpatch_level_2_split_valid_y.h5', 'r') as f:
        valY = f['y'][()]
    with h5py.File('pcam_data/camelyonpatch_level_2_split_test_x.h5', 'r') as f:
        testX = f['x'][()]
    with h5py.File('pcam_data/camelyonpatch_level_2_split_test_y.h5', 'r') as f:
        testY = f['y'][()]
        
    if K.image_data_format() == 'channels_first':
        raise NotImplementedError()

    return (trainX, trainY), (valX, valY), (testX, testY)


##############################################################################
"""RANDOM SAMPLE SELECTION"""
##############################################################################
def random_sample_selection(pos_ind_training, neg_ind_training, number_of_samples, WITNESS_RATIO):
    # If consistent values are desired, use random.seed(incremental_seed_number) before every random generation
    random_positives = random.sample(pos_ind_training, int(np.floor(number_of_samples*WITNESS_RATIO)))
    random_negatives = random.sample(neg_ind_training, int(np.ceil(number_of_samples*(1-WITNESS_RATIO))))
    randomlist = random_positives + random_negatives
    random.shuffle(randomlist)
    
    return randomlist
    
    
##############################################################################
"""MNIST GENERATE LABEL INFO FOR EXPERIMENT 3"""
##############################################################################
def MNIST_generate_exp_3_bag_label_info(Y, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, N_INS_THIRD_LEVEL, pos_ind_training, neg_ind_training, WITNESS_RATIO):
    # Predefine as a negative bag
    third_layer_labels = False
    
    # Randomly select samples
    randomlist = random_sample_selection(pos_ind_training, neg_ind_training, N_INS_THIRD_LEVEL, WITNESS_RATIO)
    
    # Extract labels
    extracted_labels = [Y[label] for label in randomlist]
    
    # Find out the label of the inner-bags           
    
    ###### FIRST LAYER ######
    first_layer_labels = []
    
    for i in range(N_INS_SECOND_LEVEL):
        
        # Check whether all elements in list are even numbers
        current_bag_label = 0
        for number_label in extracted_labels[i*BAG_SIZE:(i+1)*BAG_SIZE]:
            if (number_label % 2) != 0:
                current_bag_label = 1
                
        # If there is at least one odd number, Check whether all elements in list are odd numbers
        if current_bag_label == 1:
            for number_label in extracted_labels[i*BAG_SIZE:(i+1)*BAG_SIZE]:
                # If there is a mix of even and odd numbers
                if (number_label % 2) == 0:
                    current_bag_label = 2
        
        first_layer_labels.extend([current_bag_label])
        
    ###### SECOND LAYER ######
    second_layer_labels = []
    
    for i in range(N_INS_FIRST_LEVEL):
        
        # Check whether all elements in list come from a bag of even numbers
        current_bag_label = 0
        for number_label in first_layer_labels[i*BAG_SIZE:(i+1)*BAG_SIZE]:
            if number_label == 1:
                current_bag_label = 1
                
        # If there is at least one odd number, Check whether all elements in list come from a bag of odd numbers
        if current_bag_label == 1:
            for number_label in first_layer_labels[i*BAG_SIZE:(i+1)*BAG_SIZE]:
                # If there is a mix of even and odd numbers
                if number_label == 0:
                    current_bag_label = 2
        
        second_layer_labels.extend([current_bag_label])
        
    ###### THIRD LAYER ######   
    for i in range(N_INS_FIRST_LEVEL):
        # Check the presence of any positive
        if second_layer_labels[i] == 1:
            third_layer_labels = True
    
    return third_layer_labels, randomlist, extracted_labels


##############################################################################
"""GENERATE LABEL INFO FOR EXPERIMENT 2"""
##############################################################################
def generate_exp_2_bag_label_info(Y, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, pos_ind_training, neg_ind_training, WITNESS_RATIO, pos_digit):
    # Predefine as a negative bag
    second_layer_labels = False
    
    # Randomly select samples
    randomlist = random_sample_selection(pos_ind_training, neg_ind_training, N_INS_SECOND_LEVEL, WITNESS_RATIO)
    
    # Extract labels
    extracted_labels = [Y[label] for label in randomlist]
    
    # Find out the label of the inner-bags      
    for i in range(BAG_SIZE):           
        # The presence of more than one positive instance in a bag makes it positive
        if 1 < extracted_labels[i*BAG_SIZE:(i+1)*BAG_SIZE].count(pos_digit):
            second_layer_labels = True
    
    return second_layer_labels, randomlist, extracted_labels


##############################################################################
"""GENERATE LABEL INFO FOR EXPERIMENT 1"""
##############################################################################
def generate_exp_1_bag_label_info(Y, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, pos_ind_training, neg_ind_training, WITNESS_RATIO, pos_digit, n_pos_bags):
    # Predefine as a negative bag
    second_layer_labels = False
    
    # Due to the nature of the experiment, for creating negative samples, there must be no positive instances
    if n_pos_bags == 0:
        randomlist = random.sample(neg_ind_training, N_INS_SECOND_LEVEL)
    else:
        # Randomly select samples
        randomlist = random_sample_selection(pos_ind_training, neg_ind_training, N_INS_SECOND_LEVEL, WITNESS_RATIO)
    
    # Extract labels
    extracted_labels = [Y[label] for label in randomlist]
    
    # Find out the label of the inner-bags      
    for i in range(BAG_SIZE):           
        # The presence of more than one positive instance in a bag makes it positive
        if 0 < extracted_labels[i*BAG_SIZE:(i+1)*BAG_SIZE].count(pos_digit):
            second_layer_labels = True
    
    return second_layer_labels, randomlist, extracted_labels
        

##############################################################################
"""MNIST MODEL FOR 1 LEVELS EXPERIMENT WITH ATTENTION"""
##############################################################################
class MNIST_Model_1_levels_w_Att(tf.keras.Model):
    
    def __init__(self, CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS):
        super(MNIST_Model_1_levels_w_Att,self).__init__()
        
        self.conv1 = Conv2D(CONV1_FILTERS, input_shape=(28, 28, 1), kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool1 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.conv2 = Conv2D(CONV2_FILTERS, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.flat = Flatten()
        
        self.first_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.first_att_drop = Dropout(DROPOUT)
        self.first_att_out = Dense(1, activation='sigmoid')
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        # Feature extractor
        x = self.conv1(x_input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x) 
        first_emb = self.flat(x)
                  
        # FIRST LAYER
        # Compute attention score
        first_att = self.first_att_in(first_emb)
        first_att = self.first_att_drop(first_att)
        first_att = self.first_att_out(first_att)
        
        # Attention-based aggregation
        fst_att = tf.transpose(first_att)
        fst_att = tf.nn.softmax(fst_att)
        outer_bag_emb = tf.matmul(fst_att, first_emb)
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred
    
    
##############################################################################
"""MNIST MODEL FOR 1 LEVELS EXPERIMENT WITHOUT ATTENTION"""
##############################################################################     
class MNIST_Model_1_levels_wo_Att(tf.keras.Model):
    
    def __init__(self, CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT):
        super(MNIST_Model_1_levels_wo_Att,self).__init__()
        
        self.conv1 = Conv2D(CONV1_FILTERS, input_shape=(28, 28, 1), kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool1 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.conv2 = Conv2D(CONV2_FILTERS, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.flat = Flatten()
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        # Feature extractor
        x = self.conv1(x_input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x) 
        first_emb = self.flat(x)
                
        # FIRST LAYER                       
        # Aggregate inner bags into outer bag
        # outer_bag_emb = tf.reduce_mean(first_emb, 0)
        outer_bag_emb = tf.reduce_max(first_emb, 0)
        
        # Reshape into a 1D vector
        outer_bag_emb = tf.reshape(outer_bag_emb, [1,int(outer_bag_emb.get_shape().as_list()[0])])
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred
    

##############################################################################
"""MNIST MODEL FOR 2 LEVELS EXPERIMENT WITH ATTENTION"""
##############################################################################
class MNIST_Model_2_levels_w_Att(tf.keras.Model):
    
    def __init__(self, CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS):
        super(MNIST_Model_2_levels_w_Att,self).__init__()
        
        self.conv1 = Conv2D(CONV1_FILTERS, input_shape=(28, 28, 1), kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool1 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.conv2 = Conv2D(CONV2_FILTERS, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.flat = Flatten()
        
        self.first_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.first_att_drop = Dropout(DROPOUT)
        self.first_att_out = Dense(1, activation='sigmoid')
        
        self.second_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.second_att_drop = Dropout(DROPOUT)
        self.second_att_out = Dense(1, activation='sigmoid')
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        x, first_lab = x_input
        
        # Feature extractor
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x) 
        first_emb = self.flat(x)
                  
        # FIRST LAYER
        # Compute attention score
        first_att = self.first_att_in(first_emb)
        first_att = self.first_att_drop(first_att)
        first_att = self.first_att_out(first_att)
        fst_att = tf.transpose(tf.nn.softmax(tf.transpose(first_att)))
        
        # Count number of instances per bag
        _, _, first_count = tf.unique_with_counts(tf.reshape(first_lab,[-1]))
        count = 0

        # For each bag and the number of instances in each bag
        for num_elements in first_count:
            
          # Get indices from current bag
          indices = tf.constant([[x] for x in range(count,count+num_elements)])
          bag_ins = tf.gather_nd(first_emb, indices)
          bag_att = tf.gather_nd(fst_att, indices)
          
          # Attention-based aggregation
          curr_bag = tf.matmul(tf.transpose(bag_att), bag_ins)
          
          # Stack resulting bag enbeddings
          if count == 0:
              second_emb = curr_bag
          else:
              second_emb = tf.concat([second_emb,curr_bag], 0)
          count += num_elements
          
        # SECOND LAYER
        # Compute attention score
        second_att = self.second_att_in(second_emb)
        second_att = self.second_att_drop(second_att)
        second_att = self.second_att_out(second_att)
        
        # Attention-based aggregation
        snd_att = tf.transpose(second_att)
        snd_att = tf.nn.softmax(snd_att)
        outer_bag_emb = tf.matmul(snd_att, second_emb)
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred
    
    
##############################################################################
"""MNIST MODEL FOR 2 LEVELS EXPERIMENT WITHOUT ATTENTION"""
##############################################################################     
class MNIST_Model_2_levels_wo_Att(tf.keras.Model):
    
    def __init__(self, CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT):
        super(MNIST_Model_2_levels_wo_Att,self).__init__()
        
        self.conv1 = Conv2D(CONV1_FILTERS, input_shape=(28, 28, 1), kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool1 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.conv2 = Conv2D(CONV2_FILTERS, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.flat = Flatten()
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        x, first_lab = x_input
        
        # Feature extractor
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x) 
        first_emb = self.flat(x)
                  
        # FIRST LAYER
        # Count number of instances per bag
        _, _, first_count = tf.unique_with_counts(tf.reshape(first_lab,[-1]))
        count = 0

        # For each bag and the number of instances in each bag
        for num_elements in first_count:
            
          # Get indices from current bag
          indices = tf.constant([[x] for x in range(count,count+num_elements)])
          bag_ins = tf.gather_nd(first_emb, indices)
          
          # Apply either mean or max to aggregate the embeddings
          curr_bag = tf.reduce_mean(bag_ins, 0)
          # curr_bag = tf.reduce_max(bag_ins, 0)
          
          # Transform it into a 1D vector
          curr_bag = tf.reshape(curr_bag, [1,int(curr_bag.get_shape().as_list()[0])])
          
          # Stack resulting bag enbeddings
          if count == 0:
              second_emb = curr_bag
          else:
              second_emb = tf.concat([second_emb,curr_bag], 0)
          count += num_elements 
                
        # SECOND LAYER                       
        # Aggregate inner bags into outer bag
        # outer_bag_emb = tf.reduce_mean(second_emb, 0)
        outer_bag_emb = tf.reduce_max(second_emb, 0)
        
        # Reshape into a 1D vector
        outer_bag_emb = tf.reshape(outer_bag_emb, [1,int(outer_bag_emb.get_shape().as_list()[0])])
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred


##############################################################################
"""MNIST MODEL FOR 3 LEVELS EXPERIMENT WITH ATTENTION"""
##############################################################################
class MNIST_Model_3_levels_w_Att(tf.keras.Model):
    
    def __init__(self, CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS):
        super(MNIST_Model_3_levels_w_Att,self).__init__()
        
        self.conv1 = Conv2D(CONV1_FILTERS, input_shape=(28, 28, 1), kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool1 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.conv2 = Conv2D(CONV2_FILTERS, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.flat = Flatten()
        
        self.first_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.first_att_drop = Dropout(DROPOUT)
        self.first_att_out = Dense(1, activation='sigmoid')
        
        self.second_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.second_att_drop = Dropout(DROPOUT)
        self.second_att_out = Dense(1, activation='sigmoid')
        
        self.third_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.third_att_drop = Dropout(DROPOUT)
        self.third_att_out = Dense(1, activation='sigmoid')
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        x, second_lab, first_lab = x_input
        
        # Feature extractor
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x) 
        first_emb = self.flat(x)
                  
        # FIRST LAYER
        # Compute attention score
        first_att = self.first_att_in(first_emb)
        first_att = self.first_att_drop(first_att)
        first_att = self.first_att_out(first_att)
        fst_att = tf.transpose(tf.nn.softmax(tf.transpose(first_att)))
        
        # Count number of instances per bag
        _, _, first_count = tf.unique_with_counts(tf.reshape(first_lab,[-1]))
        count = 0

        # For each bag and the number of instances in each bag
        for num_elements in first_count:
            
          # Get indices from current bag
          indices = tf.constant([[x] for x in range(count,count+num_elements)])
          bag_ins = tf.gather_nd(first_emb, indices)
          bag_att = tf.gather_nd(fst_att, indices)
          
          # Attention-based aggregation
          curr_bag = tf.matmul(tf.transpose(bag_att), bag_ins)
          
          # Stack resulting bag enbeddings
          if count == 0:
              second_emb = curr_bag
          else:
              second_emb = tf.concat([second_emb,curr_bag], 0)
          count += num_elements
          
        # SECOND LAYER
        # Compute attention score
        second_att = self.second_att_in(second_emb)
        second_att = self.second_att_drop(second_att)
        second_att = self.second_att_out(second_att)
        snd_att = tf.transpose(tf.nn.softmax(tf.transpose(second_att)))
        
        # Count number of instances per bag
        _, _, second_count = tf.unique_with_counts(tf.reshape(second_lab,[-1]))
        count = 0

        # For each bag and the number of instances in each bag
        for num_elements in second_count:
          # Get indices from current bag
          indices = tf.constant([[x] for x in range(count,count+num_elements)])
          bag_ins = tf.gather_nd(second_emb, indices)
          bag_att = tf.gather_nd(snd_att, indices)
          
          # Attention-based aggregation
          curr_bag = tf.matmul(tf.transpose(bag_att), bag_ins)
          
          # Stack resulting bag enbeddings
          if count == 0:
              third_emb = curr_bag
          else:
              third_emb = tf.concat([third_emb,curr_bag], 0)
          count += num_elements  
                
        # THIRD LAYER
        # Compute attention score
        third_att = self.third_att_in(third_emb)
        third_att = self.third_att_drop(third_att)
        third_att = self.third_att_out(third_att)
        
        # Attention-based aggregation
        trd_att = tf.transpose(third_att)
        trd_att = tf.nn.softmax(trd_att)
        outer_bag_emb = tf.matmul(trd_att, third_emb)
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred
    
    
##############################################################################
"""MNIST MODEL FOR 3 LEVELS EXPERIMENT WITHOUT ATTENTION"""
##############################################################################     
class MNIST_Model_3_levels_wo_Att(tf.keras.Model):
    
    def __init__(self, CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT):
        super(MNIST_Model_3_levels_wo_Att,self).__init__()
        
        self.conv1 = Conv2D(CONV1_FILTERS, input_shape=(28, 28, 1), kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool1 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.conv2 = Conv2D(CONV2_FILTERS, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(POOLING_SIZE, POOLING_SIZE))
        self.flat = Flatten()
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        x, second_lab, first_lab = x_input
        
        # Feature extractor
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x) 
        first_emb = self.flat(x)
                  
        # FIRST LAYER
        # Count number of instances per bag
        _, _, first_count = tf.unique_with_counts(tf.reshape(first_lab,[-1]))
        count = 0

        # For each bag and the number of instances in each bag
        for num_elements in first_count:
            
          # Get indices from current bag
          indices = tf.constant([[x] for x in range(count,count+num_elements)])
          bag_ins = tf.gather_nd(first_emb, indices)
          
          # Apply either mean or max to aggregate the embeddings
          curr_bag = tf.reduce_mean(bag_ins, 0)
          # curr_bag = tf.reduce_max(bag_ins, 0)
          
          # Transform it into a 1D vector
          curr_bag = tf.reshape(curr_bag, [1,int(curr_bag.get_shape().as_list()[0])])
          
          # Stack resulting bag enbeddings
          if count == 0:
              second_emb = curr_bag
          else:
              second_emb = tf.concat([second_emb,curr_bag], 0)
          count += num_elements
          
        # SECOND LAYER
        # Count number of instances per bag
        _, _, second_count = tf.unique_with_counts(tf.reshape(second_lab,[-1]))
        count = 0

        # For each bag and the number of instances in each bag
        for num_elements in second_count:
            
          # Get indices from current bag
          indices = tf.constant([[x] for x in range(count,count+num_elements)])
          bag_ins = tf.gather_nd(second_emb, indices)
          
          # Apply either mean or max to aggregate the embeddings
          curr_bag = tf.reduce_mean(bag_ins, 0)
          # curr_bag = tf.reduce_max(bag_ins, 0)
          
          # Reshape into a 1D vector
          curr_bag = tf.reshape(curr_bag, [1,int(curr_bag.get_shape().as_list()[0])])
          
          # Stack resulting bag enbeddings
          if count == 0:
              third_emb = curr_bag
          else:
              third_emb = tf.concat([third_emb,curr_bag], 0)
          count += num_elements  
                
        # THIRD LAYER                       
        # Aggregate inner bags into outer bag
        # outer_bag_emb = tf.reduce_mean(third_emb, 0)
        outer_bag_emb = tf.reduce_max(third_emb, 0)
        
        # Reshape into a 1D vector
        outer_bag_emb = tf.reshape(outer_bag_emb, [1,int(outer_bag_emb.get_shape().as_list()[0])])
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred
   

##############################################################################
"""PCAM MODEL FOR 1 LEVELS EXPERIMENT WITH ATTENTION"""
##############################################################################
class PCAM_Model_1_levels_w_Att(tf.keras.Model):
    
    def __init__(self, CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS):
        super(PCAM_Model_1_levels_w_Att,self).__init__()
        
        self.base_model = VGG16(include_top=False, weights='imagenet', pooling='max')
        
        self.first_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.first_att_drop = Dropout(DROPOUT)
        self.first_att_out = Dense(1, activation='sigmoid')
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        # Feature extractor
        first_emb = self.base_model(x_input)
                  
        # FIRST LAYER
        # Compute attention score
        first_att = self.first_att_in(first_emb)
        first_att = self.first_att_drop(first_att)
        first_att = self.first_att_out(first_att)
        
        # Attention-based aggregation
        fst_att = tf.transpose(first_att)
        fst_att = tf.nn.softmax(fst_att)
        outer_bag_emb = tf.matmul(fst_att, first_emb)
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred
    
    
##############################################################################
"""PCAM MODEL FOR 1 LEVELS EXPERIMENT WITHOUT ATTENTION"""
##############################################################################     
class PCAM_Model_1_levels_wo_Att(tf.keras.Model):
    
    def __init__(self,CLASSIFIER_UNITS, DROPOUT):
        super(PCAM_Model_1_levels_wo_Att,self).__init__()
        
        self.base_model = VGG16(include_top=False, weights='imagenet', pooling='max')
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        # Feature extractor
        first_emb = self.base_model(x_input)
                
        # FIRST LAYER                       
        # Aggregate inner bags into outer bag
        # outer_bag_emb = tf.reduce_mean(first_emb, 0)
        outer_bag_emb = tf.reduce_max(first_emb, 0)
        
        # Reshape into a 1D vector
        outer_bag_emb = tf.reshape(outer_bag_emb, [1,int(outer_bag_emb.get_shape().as_list()[0])])
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred
    

##############################################################################
"""PCAM MODEL FOR 2 LEVELS EXPERIMENT WITH ATTENTION"""
##############################################################################
class PCAM_Model_2_levels_w_Att(tf.keras.Model):
    
    def __init__(self,CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS):
        super(PCAM_Model_2_levels_w_Att,self).__init__()
        
        self.base_model = VGG16(include_top=False, weights='imagenet', pooling='max')
        
        self.first_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.first_att_drop = Dropout(DROPOUT)
        self.first_att_out = Dense(1, activation='sigmoid')
        
        self.second_att_in = Dense(ATTENTION_UNITS, activation='tanh')
        self.second_att_drop = Dropout(DROPOUT)
        self.second_att_out = Dense(1, activation='sigmoid')
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        x, first_lab = x_input
        
        # Feature extractor
        first_emb = self.base_model(x_input)
                  
        # FIRST LAYER
        # Compute attention score
        first_att = self.first_att_in(first_emb)
        first_att = self.first_att_drop(first_att)
        first_att = self.first_att_out(first_att)
        fst_att = tf.transpose(tf.nn.softmax(tf.transpose(first_att)))
        
        # Count number of instances per bag
        _, _, first_count = tf.unique_with_counts(tf.reshape(first_lab,[-1]))
        count = 0

        # For each bag and the number of instances in each bag
        for num_elements in first_count:
            
          # Get indices from current bag
          indices = tf.constant([[x] for x in range(count,count+num_elements)])
          bag_ins = tf.gather_nd(first_emb, indices)
          bag_att = tf.gather_nd(fst_att, indices)
          
          # Attention-based aggregation
          curr_bag = tf.matmul(tf.transpose(bag_att), bag_ins)
          
          # Stack resulting bag enbeddings
          if count == 0:
              second_emb = curr_bag
          else:
              second_emb = tf.concat([second_emb,curr_bag], 0)
          count += num_elements
          
        # SECOND LAYER
        # Compute attention score
        second_att = self.second_att_in(second_emb)
        second_att = self.second_att_drop(second_att)
        second_att = self.second_att_out(second_att)
        
        # Attention-based aggregation
        snd_att = tf.transpose(second_att)
        snd_att = tf.nn.softmax(snd_att)
        outer_bag_emb = tf.matmul(snd_att, second_emb)
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred
    
    
##############################################################################
"""PCAM MODEL FOR 2 LEVELS EXPERIMENT WITHOUT ATTENTION"""
##############################################################################     
class PCAM_Model_2_levels_wo_Att(tf.keras.Model):
    
    def __init__(self, CLASSIFIER_UNITS, DROPOUT):
        super(PCAM_Model_2_levels_wo_Att,self).__init__()
        
        self.base_model = VGG16(include_top=False, weights='imagenet', pooling='max')
        
        self.classifier_in = Dense(CLASSIFIER_UNITS)
        self.classifier_drop = Dropout(DROPOUT)
        self.classifier_out = Dense(1, activation='sigmoid')
        
    def call(self,x_input):
        
        x, first_lab = x_input
        
        # Feature extractor
        first_emb = self.base_model(x_input)
                  
        # FIRST LAYER
        # Count number of instances per bag
        _, _, first_count = tf.unique_with_counts(tf.reshape(first_lab,[-1]))
        count = 0

        # For each bag and the number of instances in each bag
        for num_elements in first_count:
            
          # Get indices from current bag
          indices = tf.constant([[x] for x in range(count,count+num_elements)])
          bag_ins = tf.gather_nd(first_emb, indices)
          
          # Apply either mean or max to aggregate the embeddings
          curr_bag = tf.reduce_mean(bag_ins, 0)
          # curr_bag = tf.reduce_max(bag_ins, 0)
          
          # Transform it into a 1D vector
          curr_bag = tf.reshape(curr_bag, [1,int(curr_bag.get_shape().as_list()[0])])
          
          # Stack resulting bag enbeddings
          if count == 0:
              second_emb = curr_bag
          else:
              second_emb = tf.concat([second_emb,curr_bag], 0)
          count += num_elements 
                
        # SECOND LAYER                       
        # Aggregate inner bags into outer bag
        # outer_bag_emb = tf.reduce_mean(second_emb, 0)
        outer_bag_emb = tf.reduce_max(second_emb, 0)
        
        # Reshape into a 1D vector
        outer_bag_emb = tf.reshape(outer_bag_emb, [1,int(outer_bag_emb.get_shape().as_list()[0])])
        
        # PREDICTION
        pred = self.classifier_in(outer_bag_emb)
        pred = self.classifier_drop(pred)
        pred = self.classifier_out(pred)
        
        return pred


##############################################################################
"""DATA GENERATOR FOR 1 LEVEL EXPERIMENT"""
##############################################################################    
class generator_1_level(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, images, bag, shuffle):
        """Initialization.

        Args:
            images: A list of bags of images.
            bag: A list of bag-of-bags labels.
            shuffle: Whether to shuffle samples.
        """
        self.images = images
        self.bag = bag
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return len(self.images)

    def __getitem__(self, index):
        """Generate one batch of data."""            
        
        # Generate bags and labels
        array_of_img, outer_bag_label = self.__data_generation(self.images[index], self.bag[index])

        return array_of_img, outer_bag_label

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.images))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, images_temp, bag_temp):
        """Generates data containing batch_size samples."""
        
        # OUTER BAG LABEL
        if bag_temp:
            outer_bag_label = np.ones((1,1))
        else:
            outer_bag_label = np.zeros((1,1))

        return np.array(images_temp, dtype=np.float32)/255, outer_bag_label


##############################################################################
"""DATA GENERATOR FOR 2 LEVELS EXPERIMENT"""
##############################################################################    
class generator_2_levels(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, images, bag, bag_size, shuffle):
        """Initialization.

        Args:
            images: A list of bags of images.
            bag: A list of bag-of-bags labels.
            bag_size: Size of the bags.
            shuffle: Whether to shuffle samples.
        """
        self.images = images
        self.bag_size = bag_size
        self.bag = bag
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return len(self.images)

    def __getitem__(self, index):
        """Generate one batch of data."""            
        
        # Generate bags and labels
        [array_of_img, first_layer_labels], outer_bag_label = self.__data_generation(self.images[index], self.bag[index])

        return [array_of_img, first_layer_labels], outer_bag_label

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.images))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, images_temp, bag_temp):
        """Generates data containing batch_size samples."""
        
        # Generate a bag_id for each instance at each level 
        
        # FIRST LAYER
        first_layer_labels = []
        for i in range(self.bag_size):
            first_layer_labels += self.bag_size * [i]                  
        first_layer_labels = np.array(first_layer_labels)
        
        # OUTER BAG LABEL
        if bag_temp:
            outer_bag_label = np.ones((1,1))
        else:
            outer_bag_label = np.zeros((1,1))

        return [np.array(images_temp, dtype=np.float32)/255, first_layer_labels], outer_bag_label
    
    
##############################################################################
"""DATA GENERATOR FOR 3 LEVELS EXPERIMENT"""
##############################################################################    
class generator_3_levels(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, images, bag, bag_size, shuffle):
        """Initialization.

        Args:
            images: A list of bags of images.
            bag: A list of bag-of-bags labels.
            bag_size: Size of the bags.
            shuffle: Whether to shuffle samples.
        """
        self.images = images
        self.bag_size = bag_size
        self.bag = bag
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return len(self.images)

    def __getitem__(self, index):
        """Generate one batch of data."""            
        
        # Generate bags and labels
        [array_of_img, second_layer_labels, first_layer_labels], outer_bag_label = self.__data_generation(self.images[index], self.bag[index])

        return [array_of_img, second_layer_labels, first_layer_labels], outer_bag_label

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.images))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, images_temp, bag_temp):
        """Generates data containing batch_size samples."""
        
        # Generate a bag_id for each instance at each level 
        
        # FIRST LAYER
        first_layer_labels = []
        for i in range(self.bag_size**2):
            first_layer_labels += self.bag_size * [i]                  
        first_layer_labels = np.array(first_layer_labels)
        
        # SECOND LAYER
        second_layer_labels = []
        for i in range(self.bag_size):
            second_layer_labels += self.bag_size * [i]                  
        second_layer_labels = np.array(second_layer_labels)
        
        # OUTER BAG LABEL
        if bag_temp:
            outer_bag_label = np.ones((1,1))
        else:
            outer_bag_label = np.zeros((1,1))

        return [np.array(images_temp, dtype=np.float32)/255, second_layer_labels, first_layer_labels], outer_bag_label