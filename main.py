# GPU settings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

# Tensorflow config
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Callbacks
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

# Import MNIST dataset
from keras.datasets import mnist

# Other supplementary imports
import my_functions
import numpy as np


if __name__ == '__main__':
    
    ##### MODEL HYPERPARAMETERS #####
    DATASET_TO_USE       = 'MNIST' # Choices are 'MNIST' or 'PCAM'
    EXPERIMENT_ID        = 1 # Choices are 1, 2, 3
    NESTED_MODEL         = True # True = NMIL / False = MIL
    ATTENTION_MODEL      = True # True = Attention-based aggregation model / False = mean_max-based aggregation model
    EPOCHS               = 20
    LOSS_FUNCTION        = ['binary_crossentropy']
    METRICS              = ['accuracy']   
    EARLY_STOP_PATIENCE  = 20 # Number of epochs without an improvement in val_loss until the model stops training
    REDUCE_LR_PATIENCE   = 10 # Number of epochs without an improvement in val_loss until the learning rate is decreased
    RESTORE_BEST_WEIGHTS = True # True = Restore weights from epoch with best val_loss / False = Restore last epoch weights  
    OPTIMIZER            = 'SGD' # Available: 'adam', 'adamax', 'nadam', 'adadelta', 'adagrad', 'SGD'.
    LR                   = 0.0002 # Learning rate
    ATTENTION_UNITS      = 256 # Number of units at attention modules
    CLASSIFIER_UNITS     = 1024 # Number of units at classifier module
    DROPOUT              = 0.2 # Dropout rate
    # MNIST FEATURE EXTRACTOR #
    CONV1_FILTERS        = 32 # Number of filters for the first convolutional layer
    CONV2_FILTERS        = 64 # Number of filters for the first convolutional layer
    KERNEL_SIZE          = 5 # Kernel size
    POOLING_SIZE         = 2 # Pooling size
    
    ##### SETTINGS AND PARAMETERS #####
    MNIST_POS_DIGIT      = 9 # Digit to be considered as the positive instance for MNIST
    PCAM_POS_CLASS       = 1 # 1 = Presence of lymph node metastasis / 0 = Others
    N_TRAIN_BAGS         = 10000 # Number of training bags
    N_POS_TRAIN_BAGS     = int(N_TRAIN_BAGS/2) # Number of positive training bags
    N_NEG_TRAIN_BAGS     = int(N_TRAIN_BAGS/2) # Number of negative training bags
    N_VAL_BAGS           = 10000 # Number of validation bags. Only applies to PCAM
    N_POS_VAL_BAGS       = int(N_VAL_BAGS/2) # Number of positive validation bags
    N_NEG_VAL_BAGS       = int(N_VAL_BAGS/2) # Number of negative validation bags
    N_TEST_BAGS          = 10000 # Number of test bags
    N_POS_TEST_BAGS      = int(N_TEST_BAGS/2) # Number of positive test bags
    N_NEG_TEST_BAGS      = int(N_TEST_BAGS/2) # Number of negative test bags
    BAG_SIZE             = 4 # Bag size at the innermost level. Current script will set subsequent inner-bags in upper levels to have the same size
    N_INS_FIRST_LEVEL    = BAG_SIZE**1 # Number of instances a bag in the first level encapsulates
    N_INS_SECOND_LEVEL   = BAG_SIZE**2 # Number of instances a bag in the second level encapsulates
    N_INS_THIRD_LEVEL    = BAG_SIZE**3 # Number of instances a bag in the third level encapsulates
    WITNESS_RATIO        = 0.5 # Ratio of positive instances in a bag (from 0.0 to 1.0)
    
    ##### FOLDERS AND FILENAMES #####
    WEIGHTS_FOLDER       = 'weights/'
    PREDICTIONS_FOLDER   = 'predictions/'
    FILENAME_HEADER      = '{}/exp_{}'.format(DATASET_TO_USE, str(EXPERIMENT_ID))
    
    
    # Set resulting paths and filenames
    predictions_filename = PREDICTIONS_FOLDER + FILENAME_HEADER
    weights_filename = WEIGHTS_FOLDER + FILENAME_HEADER
    
    if NESTED_MODEL:
        predictions_filename = predictions_filename + '_NMIL'
        weights_filename = weights_filename + '_NMIL'
    else:
        predictions_filename = predictions_filename + '_MIL'
        weights_filename = weights_filename + '_MIL' 
        
    if ATTENTION_MODEL:
        predictions_filename = predictions_filename + '_w_att.obj'
        weights_filename = weights_filename + '_w_att'
    else:
        predictions_filename = predictions_filename + '_wo_att.obj'
        weights_filename = weights_filename + '_wo_att'

    # Load corresponding dataset
    if DATASET_TO_USE == 'MNIST':
        # Load MNIST dataset
        (trainX, trainY), (testX, testY) = mnist.load_data() # Test set will be used both as validation and test set
        trainX = trainX[..., np.newaxis]
        testX = testX[..., np.newaxis]   
        pos_digit = MNIST_POS_DIGIT
    elif DATASET_TO_USE == 'PCAM':
        # Load PCAM dataset
        (trainX, trainY), (valX, valY), (testX, testY) = my_functions.load_PCAM()
        trainY = trainY.reshape((len(trainY),))
        valY = valY.reshape((len(valY),))
        testY = testY.reshape((len(testY),))
        pos_digit = PCAM_POS_CLASS
        
    # Indexes of positive and negative intances
    if EXPERIMENT_ID in [1, 2]:
        #For this experiment, digit 9 is the positive instance and all other digits are negatives
        pos_ind_training = list(np.where(trainY == pos_digit)[0])
        neg_ind_training = list(np.where(trainY != pos_digit)[0])
        if DATASET_TO_USE == 'PCAM':
            pos_ind_val = list(np.where(valY == pos_digit)[0])
            neg_ind_val = list(np.where(valY != pos_digit)[0])
        pos_ind_test = list(np.where(testY == pos_digit)[0])
        neg_ind_test = list(np.where(testY != pos_digit)[0])
        len_of_array = N_INS_SECOND_LEVEL
    elif EXPERIMENT_ID == 3:
        #For this experiment, positives are odd numbers and negatives are even numbers
        pos_ind_training = list(np.where((trainY % 2) != 0)[0])
        neg_ind_training = list(np.where((trainY % 2) == 0)[0])
        pos_ind_test = list(np.where((testY % 2) != 0)[0])
        neg_ind_test = list(np.where((testY % 2) == 0)[0])
        len_of_array = N_INS_THIRD_LEVEL
    
    ### DEFINE TRAINING SET ###
    
    # Generate bags and save them in dictionaries
    train_data = dict()
    train_labels = dict()
    train_bag = dict()
    indexes = len(trainX)
    index = 0
    
    # Keep generating bags until the number of positive and negative bags is met        
    while(N_POS_TRAIN_BAGS > 0 or N_NEG_TRAIN_BAGS > 0):
        
        # Initialize array of images
        array_of_imgs = np.zeros((len_of_array, trainX[0].shape[0], trainX[0].shape[1], trainX[0].shape[2]))
        
        # Generate bag label info
        if EXPERIMENT_ID == 1:
            final_label, randomlist, extracted_labels = my_functions.generate_exp_1_bag_label_info(trainY, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, pos_ind_training, neg_ind_training, WITNESS_RATIO, pos_digit, N_POS_TRAIN_BAGS)
        elif EXPERIMENT_ID == 2:
            final_label, randomlist, extracted_labels = my_functions.generate_exp_2_bag_label_info(trainY, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, pos_ind_training, neg_ind_training, WITNESS_RATIO, pos_digit)
        elif EXPERIMENT_ID == 3:
            final_label, randomlist, extracted_labels = my_functions.MNIST_generate_exp_3_bag_label_info(trainY, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, N_INS_THIRD_LEVEL, pos_ind_training, neg_ind_training, WITNESS_RATIO)
        
        # Update current generation status
        # If the bag is positive and the number of positives is not met
        if final_label and N_POS_TRAIN_BAGS > 0:
            # Save data in dictionary
            for count, img_index in enumerate(randomlist):
                array_of_imgs[count] = trainX[img_index]
            train_data[index] = array_of_imgs
            train_labels[index] = extracted_labels
            train_bag[index] = final_label
            
            # Update indexes
            N_POS_TRAIN_BAGS -= 1
            index += 1 
        
        # If the bag is negative and the number of negatives is not met
        elif not final_label and N_NEG_TRAIN_BAGS > 0:
            # Save data in dictionary
            for count, img_index in enumerate(randomlist):
                array_of_imgs[count] = trainX[img_index]
            train_data[index] = array_of_imgs
            train_labels[index] = extracted_labels
            train_bag[index] = final_label
            
            # Update indexes
            N_NEG_TRAIN_BAGS -= 1
            index += 1 
      
    # PCAM does have a validation set
    if DATASET_TO_USE == 'PCAM': 
        ### DEFINE VALIDATION SET ###
            
        # Generate bags and save them in dictionaries
        val_data = dict()
        val_labels = dict()
        val_bag = dict()
        indexes = len(valX)
        index = 0
           
        # Keep generating bags until the number of positive and negative bags is met      
        while(N_POS_VAL_BAGS > 0 or N_NEG_VAL_BAGS > 0):
            
            # Initialize array of images
            array_of_imgs = np.zeros((len_of_array, valX[0].shape[0], valX[0].shape[1], valX[0].shape[2]))
            
            # Generate bag label info
            if EXPERIMENT_ID == 1:
                final_label, randomlist, extracted_labels = my_functions.generate_exp_1_bag_label_info(valY, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, pos_ind_val, neg_ind_val, WITNESS_RATIO, pos_digit, N_POS_VAL_BAGS)
            elif EXPERIMENT_ID == 2:
                final_label, randomlist, extracted_labels = my_functions.generate_exp_2_bag_label_info(valY, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, pos_ind_val, neg_ind_val, WITNESS_RATIO, pos_digit)
            
            # Update current generation status     
            # If the bag is positive and the number of positives is not met
            if final_label and N_POS_VAL_BAGS > 0:
                # Save data in dictionary
                for count, img_index in enumerate(randomlist):
                    array_of_imgs[count] = valX[img_index]
                val_data[index] = array_of_imgs
                val_labels[index] = extracted_labels
                val_bag[index] = final_label
                
                # Update indexes
                N_POS_VAL_BAGS -= 1
                index += 1 
                
            # If the bag is negative and the number of negatives is not met
            elif not final_label and N_NEG_VAL_BAGS > 0:
                # Save data in dictionary
                for count, img_index in enumerate(randomlist):
                    array_of_imgs[count] = valX[img_index]
                val_data[index] = array_of_imgs
                val_bag[index] = extracted_labels
                val_bag[index] = final_label
                
                # Update indexes
                N_NEG_VAL_BAGS -= 1
                index += 1 
            
    
    ### DEFINE TEST SET ###
        
    # Generate bags and save them in dictionaries
    test_data = dict()
    test_labels = dict()
    test_bag = dict()
    indexes = len(testX)
    index = 0
       
    # Keep generating bags until the number of positive and negative bags is met      
    while(N_POS_TEST_BAGS > 0 or N_NEG_TEST_BAGS > 0):
        
        # Initialize array of images
        array_of_imgs = np.zeros((len_of_array, testX[0].shape[0], testX[0].shape[1], testX[0].shape[2]))
        
        # Generate bag label info
        if EXPERIMENT_ID == 1:
            final_label, randomlist, extracted_labels = my_functions.generate_exp_1_bag_label_info(testY, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, pos_ind_test, neg_ind_test, WITNESS_RATIO, pos_digit, N_POS_TEST_BAGS)
        elif EXPERIMENT_ID == 2:
            final_label, randomlist, extracted_labels = my_functions.generate_exp_2_bag_label_info(testY, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, pos_ind_test, neg_ind_test, WITNESS_RATIO, pos_digit)
        elif EXPERIMENT_ID == 3:
            final_label, randomlist, extracted_labels = my_functions.MNIST_generate_exp_3_bag_label_info(testY, BAG_SIZE, N_INS_FIRST_LEVEL, N_INS_SECOND_LEVEL, N_INS_THIRD_LEVEL, pos_ind_test, neg_ind_test, WITNESS_RATIO)
        
        # Update current generation status     
        # If the bag is positive and the number of positives is not met
        if final_label and N_POS_TEST_BAGS > 0:
            # Save data in dictionary
            for count, img_index in enumerate(randomlist):
                array_of_imgs[count] = testX[img_index]
            test_data[index] = array_of_imgs
            test_labels[index] = extracted_labels
            test_bag[index] = final_label
            
            # Update indexes
            N_POS_TEST_BAGS -= 1
            index += 1 
            
        # If the bag is negative and the number of negatives is not met
        elif not final_label and N_NEG_TEST_BAGS > 0:
            # Save data in dictionary
            for count, img_index in enumerate(randomlist):
                array_of_imgs[count] = testX[img_index]
            test_data[index] = array_of_imgs
            test_labels[index] = extracted_labels
            test_bag[index] = final_label
            
            # Update indexes
            N_NEG_TEST_BAGS -= 1
            index += 1 
        
    # Define generators and models
    if EXPERIMENT_ID in [1, 2] and NESTED_MODEL == False:
        train_generator = my_functions.generator_1_level(images=train_data,
                                                                bag=train_bag,
                                                                shuffle=True)
        
        test_generator = my_functions.generator_1_level(images=test_data,
                                                                bag=test_bag,
                                                                shuffle=True)
        
        # Define model
        if DATASET_TO_USE == 'MNIST':
            if ATTENTION_MODEL:
                classifier_model = my_functions.MNIST_Model_1_levels_w_Att(CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS)
            else:
                classifier_model = my_functions.MNIST_Model_1_levels_wo_Att(CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT)
                
        elif DATASET_TO_USE == 'PCAM':
            val_generator = my_functions.generator_1_level(images=val_data,
                                                                bag=val_bag,
                                                                shuffle=True)
            if ATTENTION_MODEL:
                print()
                classifier_model = my_functions.PCAM_Model_1_levels_w_Att(CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS)
            else:
                classifier_model = my_functions.PCAM_Model_1_levels_wo_Att(CLASSIFIER_UNITS, DROPOUT)
        
    elif EXPERIMENT_ID in [1, 2] and NESTED_MODEL == True:
        train_generator = my_functions.generator_2_levels(images=train_data,
                                                                bag=train_bag,
                                                                bag_size=BAG_SIZE,
                                                                shuffle=True)
        
        test_generator = my_functions.generator_2_levels(images=test_data,
                                                                bag=test_bag,
                                                                bag_size=BAG_SIZE,
                                                                shuffle=True)
                                                
        # Define model
        if DATASET_TO_USE == 'MNIST':
            if ATTENTION_MODEL:
                classifier_model = my_functions.MNIST_Model_2_levels_w_Att(CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS)
            else:
                classifier_model = my_functions.MNIST_Model_2_levels_wo_Att(CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT)
                
        elif DATASET_TO_USE == 'PCAM':
            val_generator = my_functions.generator_2_levels(images=val_data,
                                                                bag=val_bag,
                                                                bag_size=BAG_SIZE,
                                                                shuffle=True)
            if ATTENTION_MODEL:
                classifier_model = my_functions.PCAM_Model_2_levels_w_Att(CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS)
            else:
                classifier_model = my_functions.PCAM_Model_2_levels_wo_Att(CLASSIFIER_UNITS, DROPOUT)
        
    elif EXPERIMENT_ID == 3:
        train_generator = my_functions.generator_3_levels(images=train_data,
                                                                bag=train_bag,
                                                                bag_size=BAG_SIZE,
                                                                shuffle=True)
        
        test_generator = my_functions.generator_3_levels(images=test_data,
                                                                bag=test_bag,
                                                                bag_size=BAG_SIZE,
                                                                shuffle=True)
        # Define model                                        
        if ATTENTION_MODEL:
            classifier_model = my_functions.MNIST_Model_3_levels_w_Att(CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT, ATTENTION_UNITS)
        else:
            classifier_model = my_functions.MNIST_Model_3_levels_wo_Att(CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOLING_SIZE, CLASSIFIER_UNITS, DROPOUT)
            
    # Define callback settings     
    my_reduce_LR_callback = ReduceLROnPlateau(monitor='val_loss',  # quantity to be monitored
                                                factor=0.1,  # multiplier applied to the learning rate
                                                patience=REDUCE_LR_PATIENCE,
                                                verbose=1,
                                                mode='auto',
                                                min_delta=0.0001, # minimum change in the monitored quantity to qualify as an improvement
                                                cooldown=0,
                                                min_lr=0)

    early_stop_val_loss_callback = EarlyStopping(monitor='val_loss',  # quantity to be monitored
                                                 min_delta=0.000001,  # minimum change in the monitored quantity to qualify as an improvement
                                                 patience=EARLY_STOP_PATIENCE,
                                                 verbose=1,
                                                 mode='auto',
                                                 baseline=None,
                                                 restore_best_weights=RESTORE_BEST_WEIGHTS)
    
    # Group callbacks
    callback_array = [early_stop_val_loss_callback, my_reduce_LR_callback]

    # Define optimizer and learning rate
    my_optimist = my_functions.define_optimizer(OPTIMIZER, LR)
    
    # Compile model
    classifier_model.compile(optimizer=my_optimist,
                                        loss=LOSS_FUNCTION,
                                        metrics=METRICS,
                                        run_eagerly=True,
                                        experimental_run_tf_function=False)    
        
    # Train model
    if DATASET_TO_USE == 'MNIST':
        classifier_history = classifier_model.fit_generator(generator=train_generator,
                                                            steps_per_epoch=None,
                                                            epochs=EPOCHS,
                                                            verbose=1,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                                            validation_data=test_generator,
                                                            callbacks=callback_array,
                                                            shuffle=True)
    elif DATASET_TO_USE == 'PCAM':
        classifier_history = classifier_model.fit_generator(generator=train_generator,
                                                            steps_per_epoch=None,
                                                            epochs=EPOCHS,
                                                            verbose=1,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                                            validation_data=val_generator,
                                                            callbacks=callback_array,
                                                            shuffle=True)
    
    # Save weights
    classifier_model.save_weights(weights_filename)
    
    # Run a prediction and save results in a pickle object
    predict = classifier_model.predict_generator(test_generator, verbose=1)
    predictions = dict()
    predictions['Data'] = test_data
    predictions['Labels'] = test_labels
    predictions['Bag'] = test_bag
    predictions['Predict'] = predict
    my_functions.pickle_save(predictions, predictions_filename)