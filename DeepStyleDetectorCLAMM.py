'''
Created on 4 Dec 2019

@author: sarah Ismael
In this program we load the CLaMM dataset, Implementing some Image preprocessing 
and generate a deep learning model and test approach on unseen data
'''

import multiprocessing
import os
import shutil
import random
import collections
import argparse
import sys
import datetime
import platform
import time
from itertools import product

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.transform import downscale_local_mean
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, convolutional, AveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def parse_commandline_args():
    """To Run: python DeepStyleDetector.py --nrows 150 --ncols 150 --epoches 10 
    --batch_size 32 --images_dir 
    ICDAR2017_CLaMM_Training_labeled/ --output_dir results150nc150 --output_label test1 
    --test_images_dir ICDAR2017_CLaMM_task1_task3_labeled --nkernals 32 --dropout 0.25 --num_patches 100"""
    
    parser = argparse.ArgumentParser(description='Deep Learning Model')
    #image processing parameters
    parser.add_argument('--nrows', type=int, default=150, help='number of rows')
    parser.add_argument('--ncols', type=int, default=150, help='number of column')
    parser.add_argument('--nchannels', type=int, default=1, help='number of channels, default = 1')
    parser.add_argument('--white_threshold', type=int, default=1, help='minimum value to consider as white pixel (value/255)')
    parser.add_argument('--min_perentage_white_pixels', type=int, default=10, help='minimum perentage of white pixels to consider page as empty')
    parser.add_argument('--row_padding', type=float, default=0.20, help='padding (percentage) input image (applied to right,top,left,right)')
    parser.add_argument('--col_padding', type=float, default=0.25, help='padding (percentage) input image (applied to right,top,left,right)')
    parser.add_argument('--num_patches', type=int, default=50, help='num_patches from each image')
    parser.add_argument('--num_cores', type=int, default=12, help='num_cores')
    
    #deep learning params
    parser.add_argument('--epoches', type=int, default=10, help='number epoches')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--nkernals', type=int, default=32, help='number kernels')
    parser.add_argument('--dropout', type=float, default=0.25, help='number kernels')
    parser.add_argument('--dense_unit', type=int, default=128, help='unit size for dense layer')
    
    #data reading params
    parser.add_argument('--skip_read_existing_arrays', action='store_const', const=True, help='')
    parser.add_argument('--images_dir', default = 'train_images_1Apr', help='')
    parser.add_argument('--test_images_dir', default = 'test_images_1Apr', help='')
    
    parser.add_argument('--output_dir', default = str(datetime.datetime.now()).replace(" ", "").replace(":", ""), help='output directory path')
    parser.add_argument('--output_label', default = "test", help='output label')
    
    parser.add_argument('--testing', action='store_const', const=True, help='')
    
    return parser.parse_args(sys.argv[1:])

args= parse_commandline_args()

def is_image_empty(input_image, white_threshold, min_perentage_white_pixels):
    
    ''' this function try to find empty patches by checking white and black pixels
    if percentage_white_pixels of a patch was equal or less than min_percentage_white_pixel 
    we keep the patch otherwise the patch neglected'''
    flattened_array = np.ndarray.flatten(input_image)
    value_counts  = collections.Counter(flattened_array)
    number_white_pixels = 0.0
    number_black_pixels = 0.0
    
    for value in value_counts:
        if value >= white_threshold:
            number_white_pixels+=value_counts[value]
        else:
            number_black_pixels+=value_counts[value]
    try:
        percentage_white_pixels = (number_white_pixels/(number_white_pixels+number_black_pixels))*100.0
    except ZeroDivisionError:
        percentage_white_pixels=0.0
    if  percentage_white_pixels >= min_perentage_white_pixels:
        return False
    else:
        return True

def preprocess_images(images, output_dir):
    
    print('Process', len(images), ' Images')
    
    for i, image in enumerate(images):
        if not image.endswith('.tif'):
            continue
        image_file = args.images_dir + '/' + image
        if 'test_data' in output_dir:
            image_file = args.test_images_dir + '/' + image
        
        #read image using opencv2
        input_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        #in this step we remove white space around boundary
        padding_rows = int(input_image.shape[0]*args.row_padding)
        padding_cols = int(input_image.shape[1]*args.col_padding)
        input_image = input_image[padding_rows:-padding_rows, padding_cols:-padding_cols]
        #edge detection using canny function
        input_image = cv2.Canny(input_image, 100, 200)
        #downsampling images by factor 2x2
        input_image = downscale_local_mean(input_image, factors=(2,2))
        #extract labels from images' name
        label = image.split('-')[0]
        input_image=input_image.astype('float32')
        #in each image we extract maximum 100 patches randomly
        patches=extract_patches_2d(input_image,(args.nrows,args.ncols), args.num_patches)
        for idx, patch in enumerate(patches):
            print("Image number processed: ", i, idx, image)
            #checke if the patch is empty or not
            if is_image_empty(patch, args.white_threshold, args.min_perentage_white_pixels):
                continue
            
            cv2.imwrite(output_dir+label+'/'+str(idx)+'_'+image, patch)
            
    return
    

def get_x_y_from_dir(dir, labels):
    # we tried to generate array for validation images and their labels to visualize the model using TensorBoard
    x_val = []
    y_val = []
    for label in labels:
        images = os.listdir(dir+'/'+label)
        for image_file in images:
            img_arr = cv2.imread(dir+'/'+label+'/'+image_file, cv2.IMREAD_UNCHANGED)
            x_val.append(img_arr)
            y_val.append(label)
    print(x_val)
    y_val=np.array(y_val)
    x_val=np.array(x_val)
    print('x_val', x_val)
    x_val=x_val.reshape(x_val.shape[0], args.nrows , args.ncols, args.nchannels)
    
    np.save(args.output_dir+'/Y_val.npy', y_val)
    np.save(args.output_dir+'/X_val.npy', x_val)
    
    return x_val, y_val
    
def preprocess_and_seperate(labels):
    empty_dir = args.output_dir+'/empty_patches/'
    train_dir = args.output_dir+'/train_data/'
    validation_dir = args.output_dir+'/validation_data/'
    test_dir = args.output_dir+'/test_data/'
    
    if os.path.isfile(args.output_dir+'/Y_val.npy') and os.path.isfile(args.output_dir+'/X_val.npy'):
        y_val = np.load(args.output_dir+'/Y_val.npy')
        x_val = np.load(args.output_dir+'/X_val.npy')
    elif os.path.isdir(validation_dir):
        x_val, y_val = get_x_y_from_dir(validation_dir, labels)
    
    try:
        os.makedirs(empty_dir)
        os.makedirs(train_dir)
        os.makedirs(validation_dir)
        os.makedirs(test_dir)
    except FileExistsError:
        print("Data directories already exist:", train_dir,validation_dir,test_dir)
        return train_dir, validation_dir, test_dir, x_val, y_val
    
    for label in labels:
        try:
            os.makedirs(train_dir+label)
            os.makedirs(validation_dir+label)
            os.makedirs(test_dir+label)
        except FileExistsError:
            pass
    
    num_images_per_thread = 50
    image_names = os.listdir(args.images_dir)
    random.shuffle(image_names)
    if args.testing:
        image_names=image_names[0:100]
        num_images_per_thread = 20
    
    train_images = image_names[0:int(len(image_names)*0.9)]
    validation_images = image_names[int(len(image_names)*0.9)::]
    
    test_images = os.listdir(args.test_images_dir)
    if args.testing:
        test_images=test_images[0:100]
    
    #test_images = image_names[int(len(image_names)*0.8)::]
    print('train_images: ', len(train_images))
    print('validation_images: ', len(validation_images))
    print('test_images: ', len(test_images))
    
    #multi-thread for train images
    train_image_lists = []
    for i in range(0, len(train_images), num_images_per_thread):
        train_image_lists.append(train_images[i:i+num_images_per_thread])
        print('Iteration:', i, ':', len(train_image_lists))
    
    if platform.system()=="Darwin": #otherwise cv2.Canny hangs when ran in parallel on Mac
        cv2.setNumThreads(0) 
    
    pool = multiprocessing.Pool(args.num_cores)
    pool.starmap(preprocess_images, product(train_image_lists, [train_dir]))
    
    #multi-thread for validation images
    validation_image_lists = []
    for i in range(0, len(validation_images), num_images_per_thread):
        validation_image_lists.append(validation_images[i:i+num_images_per_thread])
        print('Iteration:', i, ':', len(validation_image_lists))
    
    if platform.system()=="Darwin": #otherwise cv2.Canny hangs when ran in parallel on Mac
        cv2.setNumThreads(0) 
    
    pool = multiprocessing.Pool(args.num_cores)
    pool.starmap(preprocess_images, product(validation_image_lists, [validation_dir]))
    #write validation patches and labels to arrays
    x_val, y_val = get_x_y_from_dir(validation_dir, labels)
    #multi-thread for test images
    test_image_lists = []
    for i in range(0, len(test_images), num_images_per_thread):
        test_image_lists.append(test_images[i:i+num_images_per_thread])
        print('Iteration:', i, ':', len(test_image_lists))
    
    if platform.system()=="Darwin": #otherwise cv2.Canny hangs when ran in parallel on Mac
        cv2.setNumThreads(0) 
    
    pool = multiprocessing.Pool(args.num_cores)
    pool.starmap(preprocess_images, product(test_image_lists, [test_dir]))
    
    
    return train_dir,validation_dir,test_dir,x_val,y_val

def load_traindata(train_dir, validation_dir, labels, y_val):
    """load data into keras Generators"""
    train_data = ImageDataGenerator()
    validation_data = ImageDataGenerator()
    
    train_generator = train_data.flow_from_directory(
                                                    directory = train_dir, 
                                                    target_size = (args.nrows, args.ncols),
                                                    color_mode = 'grayscale',
                                                    batch_size=args.batch_size,
                                                    classes = labels,
                                                    class_mode='categorical',
                                                    shuffle=True)
    
    print('number of train_files in ', train_dir, len(train_generator.filenames))
    
    #get y_val numpy array and replace cursiva, gotic, etc with their indices
    labels_values = train_generator.class_indices
    y_val_int = []
    for label in y_val:
        y_val_int.append(labels_values[label])
    y_val = np_utils.to_categorical(np.array(y_val_int), nclass)
    
    validation_generator = validation_data.flow_from_directory(
                                                    directory = validation_dir, 
                                                    target_size = (args.nrows, args.ncols),
                                                    color_mode = 'grayscale',
                                                    batch_size=args.batch_size,
                                                    classes = labels,
                                                    class_mode='categorical',
                                                    shuffle=True)
    print('number of validation_files in ', validation_dir, len(validation_generator.filenames))
    
    return train_generator,validation_generator,y_val

def load_testdata(test_dir):
    """load data into keras generators"""
    test_data = ImageDataGenerator()
    test_generator = test_data.flow_from_directory(
                                               directory = test_dir,
                                               target_size = (args.nrows, args.ncols),
                                               color_mode = 'grayscale',
                                               batch_size=1,
                                               class_mode=None,
                                               shuffle=False)
    print('number of test_files in ', test_dir, len(test_generator.filenames))
    return test_generator

def GenerateTheModel(input_shape,nclass):
    '''The model consists of ten CNN layers that followed by AveragPooling and Dropout layers
    except last block of CNN which is not followed by Dropout layer and followed by three dense layers
    
    '''
    kernal_size=(3,3)#window slide
    pool_size=(2,2)#2,2
    opt=Adam(lr=0.001)#learning rule
    
    model = Sequential()
    model.add(convolutional.Conv2D(args.nkernals, kernal_size, padding='valid' ,activation='relu', 
                                   input_shape=input_shape))
    model.add(convolutional.Conv2D(args.nkernals, kernal_size, activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size, strides=2))#MaxPooling2D
    model.add(Dropout(rate=args.dropout))
    
    model.add(convolutional.Conv2D(args.nkernals*2, kernal_size, activation='relu', input_shape=input_shape))
    model.add(convolutional.Conv2D(args.nkernals*2, kernal_size, activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size, strides=2))#MaxPooling2D
    model.add(Dropout(rate=args.dropout))

    model.add(convolutional.Conv2D(args.nkernals*4, kernal_size, activation='relu', input_shape=input_shape))
    model.add(convolutional.Conv2D(args.nkernals*4, kernal_size, activation='relu', input_shape=input_shape))
    model.add(convolutional.Conv2D(args.nkernals*4, kernal_size, activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size, strides=2))#MaxPooling2D
    model.add(Dropout(rate=args.dropout))

    model.add(convolutional.Conv2D(args.nkernals*8, kernal_size, activation='relu', input_shape=input_shape))
    model.add(convolutional.Conv2D(args.nkernals*8, kernal_size, activation='relu', input_shape=input_shape))
    model.add(convolutional.Conv2D(args.nkernals*8, kernal_size, activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size, strides=2))#MaxPooling2D
    
    model.add(Flatten())
    model.add(Dense(units=args.dense_unit, activation='relu'))
    model.add(Dense(units=args.dense_unit, activation='relu'))
    model.add(Dense(units=nclass, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
    

def train_model(model, train_generator, validation_generator,epochs, x_val, y_val):
    # In this function we train the model and evaluate it in each epoch to see model performance 
    steps_per_epoch_train = train_generator.n//train_generator.batch_size
    steps_per_epoch_val = validation_generator.n//validation_generator.batch_size
    
    train_result=model.fit_generator(generator=train_generator, epochs=epochs, 
                         validation_data=validation_generator,
                         steps_per_epoch = steps_per_epoch_train,
                         validation_steps = steps_per_epoch_val,
                         verbose=1,
                         shuffle=True,
                         use_multiprocessing=True)
    
    validation_result=model.evaluate_generator(generator=validation_generator,
                                               steps=steps_per_epoch_val)
    
    return train_result, validation_result
    
def test_model(model, test_generator, labels_values):
    #prediction found by averaging probabilities for each class in all patches for that image 
    steps_per_epoch_test = test_generator.n//test_generator.batch_size
    Y_pred=model.predict_generator(test_generator,steps=steps_per_epoch_test)
    file_name_patches=test_generator.filenames
    Y_pred_patches=np.argmax(Y_pred, axis=1)
    Y_true_patches=[]
    
    dic_test_files={}
    Y_pred_images=[]
    Y_true_images=[]
    file_name_images=[]
    for i,image_name in enumerate(file_name_patches):
        image_name='_'.join(image_name.split('/')[-1].split('_')[1::])
        true_label=labels_values[image_name.split('-')[0]]
        Y_true_patches.append(true_label)
        try:
            dic_test_files[image_name].append(i)
        except KeyError:
            dic_test_files[image_name] = [i]
    
    
    for file_name in dic_test_files.keys():
        
        file_name_images.append(file_name) 
        true_label=labels_values[file_name.split('-')[0]]
        Y_true_images.append(true_label)
        
        predict_values=[]#create an empty array
        for idx in dic_test_files[file_name]:
            predict_values.append([Y_pred[idx]])
             
        sum_preds = np.sum(predict_values, axis=0)
        pred_class=sum_preds.argmax(axis=1)[0]
        Y_pred_images.append(pred_class)
        
    return Y_pred_patches, Y_true_patches, file_name_patches, Y_true_images, Y_pred_images, file_name_images

if __name__ =='__main__':
    start_time = time.time()   
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    model_file_path = args.output_dir+'/'+args.output_label+'_model.h5'
    
    values_labels= {0: 'Caroline', 1 :'Cursiva', 2:'HalfUncial', 3:'Humanistic', 
                   4:'HumanisticCursive', 5:'Hybrida', 6:'Praegothica', 7:'Semihybrida', 
                   8:'Semitextualis', 9:'SouthernTextualis', 10:'Textualis', 11:'Uncial'
                   }
    labels = list(values_labels.values())
    
    nclass=len(labels)
    input_shape=(args.nrows, args.ncols, args.nchannels)
    train_dir,validation_dir,test_dir,x_val,y_val = preprocess_and_seperate(labels)
    
    test_generator = load_testdata(test_dir)
    train_generator, validation_generator, y_val = load_traindata(train_dir, 
                                                                  validation_dir, 
                                                                  labels,
                                                                  y_val)
    results_file = open(args.output_dir+'/' + args.output_label + 'results.txt', 'a+')
    
    if not os.path.isfile(model_file_path):
        model = GenerateTheModel(input_shape,nclass)
        print(model.summary())
        train_result, validation_result = train_model(model, train_generator, 
                                                     validation_generator, 
                                                     epochs=args.epoches,
                                                     x_val=x_val, y_val=y_val)
        model.save(model_file_path)
        
        epochs_results = "Epochs:{}\nEpoch results from patches: {}\n".format(
                        train_result.epoch, train_result.history)
        
        print(epochs_results)
        print('Training result accuracy: {}; Validation accuracy {}'.format(
            list(map(lambda x: x*100, train_result.history['accuracy'])),
            list(map(lambda x: x*100, train_result.history['val_accuracy']))
            ))
        results_file.write('*******Training Results***** From: {}\n\n'.format(str(datetime.datetime.now())))
        results_file.write("Used images for training and validation in directories: {} and {}\n".format(train_dir, validation_dir))
        model.summary(print_fn=lambda x: results_file.write(x + '\n'))
        results_file.write(epochs_results+'\n')
        
    else:
        print("Loading Existing Model: ", model_file_path)
        model = load_model(model_file_path, compile = False)
    
    results_file.write('*******Test Results***** From: {}\nUsed Model: {}\n'.format(str(datetime.datetime.now()), model_file_path))
    results_file.write("Used test images in directory: {}\n".format(test_dir))
    labels_values = train_generator.class_indices
    values_labels = {}
    for label in labels_values.keys():
        value = labels_values[label]
        values_labels[value] = label
    
    Y_pred_patches, Y_true_patches, file_name_patches, Y_true_images, Y_pred_images, file_name_images = test_model(model, 
                                                                               test_generator,
                                                                               labels_values)
    
    print("labels",labels)
    
    conf_matrix_patches = confusion_matrix(Y_true_patches, Y_pred_patches)
    classif_report_patches = classification_report(Y_true_patches, Y_pred_patches)
    print('classif_report_pathces', classif_report_patches)
    
    conf_matrix_images = confusion_matrix(Y_true_images, Y_pred_images)
    classif_report_images = classification_report(Y_true_images, Y_pred_images)
    print('classif_report_images', classif_report_images)
    

    results_file.write('classif_report_patches:\n' + classif_report_patches+'\n')
    results_file.write('classif_report_images:\n' + classif_report_images+'\n')
    
    conf_matrix_patches_str = '\t' + np.array2string(np.array(range(0, len(conf_matrix_patches)))).replace('[', '').replace(']', '') + '\n'
    for i, row in enumerate(conf_matrix_patches):
        conf_matrix_patches_str+= str(i) + '\t' + np.array2string(row).replace('[', '').replace(']', '') + '\n'
    
    conf_matrix_images_str = '\t' + np.array2string(np.array(range(0, len(conf_matrix_images)))).replace('[', '').replace(']', '') + '\n'
    for i, row in enumerate(conf_matrix_images):
        conf_matrix_images_str+= str(i) + '\t' + np.array2string(row).replace('[', '').replace(']', '') + '\n'
    
    results_file.write('conf_matrix_patches:\n' + conf_matrix_patches_str +'\n')
    results_file.write('conf_matrix_images:\n' + conf_matrix_images_str +'\n')
    
    try:
        os.makedirs(args.output_dir + '/' + args.output_label + 'misclassified_images')
        os.makedirs(args.output_dir + '/' + args.output_label + 'misclassified_patches')
    except FileExistsError:
        pass
    tp=0
    fp=0
    for  i, value in enumerate(Y_pred_patches):
        if Y_pred_patches[i]==Y_true_patches[i]:
            tp+=1
        else:
            fp+=1
            '''
            shutil.copy(args.output_dir+'/test_data/'+file_name_patches[i], 
                        args.output_dir+ '/'+args.output_label+'misclassified_patches'+'/'+
                        values_labels[Y_pred_patches[i]]+'_'+values_labels[Y_true_patches[i]]+'_'+file_name_patches[i].split('/')[-1])
            '''
    accuracy_for_test_patches=(tp/(tp+fp))*100.0
    print("Accuracy Test Patches",accuracy_for_test_patches)
    
    tp=0
    fp=0
    for  i, value in enumerate(Y_pred_images):
        if Y_pred_images[i]==Y_true_images[i]:
            tp+=1
        else:
            fp+=1
            '''
            shutil.copy(args.images_dir+'/'+file_name_images[i], 
                        args.output_dir+ '/'+args.output_label+'misclassified_images'+'/'+
                        values_labels[Y_pred_images[i]]+'_'+values_labels[Y_true_images[i]]+'_'+file_name_images[i])
            '''
    accuracy_for_test=(tp/(tp+fp))*100.0
    print("Accuracy Test images",accuracy_for_test)
    
    df_cm = pd.DataFrame(conf_matrix_patches, index = [values_labels[i] for i in range(0,nclass)],
                      columns = [values_labels[i] for i in range(0,nclass)])
    
    print('Confusion matrix_patches:\n' + df_cm.to_string() + '\n')
    results_file.write('Confusion matrix_patches:\n' + df_cm.to_string() + '\n')
    
    df_cm_images = pd.DataFrame(conf_matrix_images, index = [values_labels[i] for i in range(0,nclass)],
                      columns = [values_labels[i] for i in range(0,nclass)])
    
    print('Confusion matrix_images:\n' + df_cm_images.to_string() + '\n')
    results_file.write('Confusion matrix_images:\n' + df_cm_images.to_string() + '\n')
    results_file.write("Accuracy Test images: "+str(accuracy_for_test)+'\n')
    results_file.write("\n\n*****Parameters used for this test********\n")
    results_file.write(str(args)+'\n')
    end_time = time.time()
    duration = end_time - start_time
    print('Running took:' + str(duration))
    results_file.write('Running time: '+ str(duration))
    results_file.close()
    

