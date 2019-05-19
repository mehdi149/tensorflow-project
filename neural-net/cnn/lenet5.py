#  @MEHDI149@       
#   @'''''''''''@ 
#  ''' -+- -+-'''   #
#  ''' \ _|_ /'''   #
#        \ /       #     
#        ---       #

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage

def extract_CrossValidationData(data ,PercentagePerDataSet = (80,10,10)):
    '''
    split data to training , testing and validation dataset
    according to the Percentage per data set 
    default : 60%, 20%, 20%

    '''

    if(sum(PercentagePerDataSet) != 100):
        raise ValueError('The list should contain the percentage of how much data  each dataset will contain')


    #training dataset   
    X_train = None
    Y_train = None
    #test dataset
    X_test = None
    Y_test = None
    #validation dataset 
    X_val = None
    Y_val = None 
    
    train_perc , valid_perc , test_perc = PercentagePerDataSet
    
    length_data = data['X'].shape[0]

    train_length = int( (length_data*train_perc) / 100 )
    valid_length = int( (length_data*valid_perc) / 100 )
    test_length  = int( (length_data*test_perc) / 100 )
    current_index = 0

    X_train = data['X'][current_index:int(train_length)]
    Y_train = data['Y'][current_index:int(train_length)]
    current_index += train_length

    X_val = data['X'][current_index:current_index+valid_length]
    Y_val = data['Y'][current_index:current_index+valid_length]
    current_index += valid_length

    X_test = data['X'][current_index:current_index+test_length]
    Y_test = data['Y'][current_index:current_index+test_length]

    print("training length : ",X_train.shape[0],Y_train.shape[0])
    print("validation length : ",X_val.shape[0],Y_val.shape[0])
    print("test length : ",X_test.shape[0],Y_test.shape[0])
    return { 'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test,'X_val': X_val, 'Y_val': Y_val }

'''
#open training file
with open('../train.csv' ,'r') as file_train:
    data = file_train.read().split('\n')[1:]
    #Training data
    print(len(data))
    X = np.ndarray(shape=(len(data),784),dtype = np.float32)
    Y = np.ndarray((len(data),1)) 
    index = 0
    for line in data:
        if(len(line) == 0):
            break
        fields = line.split(",")
        X[index,:] = fields[1:]
        Y[index,:] = fields[0]
        index = index + 1

    X = (X - 128 )/ 128
    print(X[0])
    X = np.resize(X,(len(data),28,28,1))

    X= np.pad(X, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    print(X[0])

    cross_val_data = extract_CrossValidationData({'X':X ,'Y':Y})

    X_train = cross_val_data['X_train']
    Y_train = cross_val_data['Y_train']
    X_test = cross_val_data['X_test']
    Y_test = cross_val_data['Y_test']
    X_val = cross_val_data['X_val']
    Y_val = cross_val_data['Y_val']

'''
def random_mini_batches(X, Y, mini_batch_size = 32, seed = 0):
    """
    Creates a list of random minibatches89 from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :,:,:]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[ k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[: , k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches   



prob = tf.placeholder_with_default(1.0, shape=())

inputs , labels =  tf.placeholder(tf.float32, shape = (None,32,32,1), name="inputs"), tf.placeholder(tf.int32, shape =(None,10), name="labels")
# conv layer nc = 6 , f=5 , p = 0 , s = 1
W1 = tf.get_variable("W1",[5,5,1,6],initializer=tf.contrib.layers.xavier_initializer(seed = 0) )
b1 = tf.get_variable("b1",(1,6),initializer=tf.zeros_initializer())
conv1 = tf.nn.conv2d(inputs ,W1, strides = [1,1,1,1] , padding = 'VALID' )
print(conv1.shape)
A1 = tf.nn.relu(conv1 + b1)
A1 = tf.nn.dropout(A1, prob)
#Layer2
#Avg Pool Layer , f = 2 , p = 0 , s = 2
pool1 = tf.nn.avg_pool(A1,ksize =[1,2,2,1],strides =[1,2,2,1],padding ='VALID')
print(pool1.shape)
#Layer3
#conv layer nc = 16 , f = 5 , p = 0 , s = 1
W2 = tf.get_variable("W2",[5,5,6,16],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
b2 = tf.get_variable("b2",(1,16),initializer=tf.zeros_initializer())
conv2 = tf.nn.conv2d(pool1 , W2, strides =[1,1,1,1] , padding = 'VALID')
A2 = tf.nn.relu(conv2 + b2)
A2 = tf.nn.dropout(A2, prob)
#Layer4
#Avg pool layer f = 2 , p =0 , s = 2
pool2 = tf.nn.avg_pool(A2, ksize =[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
fl2 = tf.contrib.layers.flatten(pool2)
#Layer5
#Fully connected layer nc = 120
fc1 = tf.contrib.layers.fully_connected(fl2 , 120 , activation_fn= tf.nn.relu)
#Layer6
#Fully connected layer nc = 84
fc2 = tf.contrib.layers.fully_connected(fc1,84,activation_fn= tf.nn.relu)

#output softmax
fc3 = tf.contrib.layers.fully_connected(fc2 ,10 ,activation_fn = None)
print("shape fc3 : ", fc3.shape)

#cost_func
cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=fc3))
print(cost.shape)
#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

'''
with tf.Session() as sess :
    saver.restore(sess,'./lenet5-model')
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    print(x_test.dtype)
    print(x_test[0])
    x_test = tf.to_float(x_test)
    print(x_test.dtype)
    x_test = (x_test - 128) /128
    print(x_test[0])
    x_test = np.resize(x_test,(x_test.shape[0],28,28,1))
    x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    y_test = tf.one_hot(y_test,10,axis = 0 )
    y_test = sess.run(y_test)
    #Y_val = tf.one_hot(Y_val,10,axis = 1 )
    #Y_val = sess.run(Y_val)
    #Y_val = Y_val.reshape(Y_val.shape[0],Y_val.shape[1])
    #Y_val = np.transpose(Y_val)

    correct_prediction = tf.equal(tf.argmax(fc3,1), tf.argmax(labels,1))
    # Calculate accuracy on the training set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc= sess.run(accuracy , feed_dict = {inputs: x_test ,labels:np.transpose(y_test)})
    print("accuracy val set : ",acc)

    label_prediction = tf.argmax(fc3 , 1)
    print("model restored ...")
    with open('../test.csv','r') as test_data:
        test_data = test_data.read()
        rows = test_data.split('\n')[1:]
        number_of_digits = len(rows)
        print(number_of_digits)
        X_kg_test = np.ndarray(shape=(number_of_digits,784),dtype = np.float32)
        cpt=0
        for row in rows:  
            X_kg_test[cpt,:] = row.split(',')
            cpt += 1

        X_kg_test = (X_kg_test - 128) / 128
        print('passsed ...')

        X_kg_test = np.resize(X_kg_test,(number_of_digits,28,28,1))
        X_kg_test = np.pad(X_kg_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        predictions = sess.run(label_prediction, feed_dict = {inputs:X_kg_test } )
        print(predictions)
        with  open('submission.csv','w') as submission_file:
            i = 0
            for prediction in predictions:
                submission_file.write(str(i+1)+','+str(prediction)+'\n')
                i += 1
'''

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


import numpy as np
import cv2

#video capture
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #draw a rectangle around the region in the frame where the image recognition will proceed 
    detection_rect = cv2.rectangle(frame, (250,250), (550,550), (0,255,0), 1)
    # Take an instant capture of the region in the frame
    frame_region = frame[300:500,300:500,:]
    #frame region to gray scale
    im_gray = cv2.cvtColor(frame_region, cv2.COLOR_RGB2GRAY);


    img28 = cv2.resize(im_gray,(28, 28), interpolation = cv2.INTER_LINEAR)

    (thresh, img28) = cv2.threshold(img28, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    while np.sum(img28[0]) == 0:
        print("on")
        img28 = img28[1:]

    while np.sum(img28[:,0]) == 0:
        img28 = np.delete(img28,0,1)

    while np.sum(img28[-1]) == 0:
        img28 = img28[:-1]

    while np.sum(img28[:,-1]) == 0:
        img28 = np.delete(img28,-1,1)

    rows,cols = img28.shape
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        img28 = cv2.resize(img28, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        img28 = cv2.resize(img28, (cols, rows))
    
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    img28 = np.lib.pad(img28,(rowsPadding,colsPadding),'constant')
    shiftx,shifty = getBestShift(img28)
    shifted = shift(img28,shiftx,shifty)
    img28 = shifted



    #preprocess the image


    # Display the resulting frame
    cv2.imshow('frame',detection_rect)
    cv2.imshow('region',im_gray)
    cv2.imshow('IMG_INFER',img28)
     

   
    
    with tf.Session() as sess:
        saver.restore(sess,'./lenet5-model')
        img28 = np.resize(img28,(1,28,28,1))
        img28 = np.pad(img28, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        pred = sess.run(tf.nn.softmax(fc3),feed_dict={inputs :img28})
        argmax = sess.run(tf.argmax(pred,1))
        if(pred[0,argmax] < 0.9):
            print("Not a number")
        else:
            print("Number prediction : ",argmax)
          

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


'''
with tf.Session() as sess :
    saver.restore(sess,'./lenet5-model')
    Y1 = tf.one_hot(Y_train,10,axis = 1 )
    Y1 = sess.run(Y1)
    Y1 = Y1.reshape(Y1.shape[0],Y1.shape[1])
    Y1 = np.transpose(Y1)
    Y_test = tf.one_hot(Y_test,10,axis = 1 )
    Y_test = sess.run(Y_test)
    Y_test = Y_test.reshape(Y_test.shape[0],Y_test.shape[1])
    Y_test = np.transpose(Y_test)
    Y_val = tf.one_hot(Y_val,10,axis = 1 )
    Y_val = sess.run(Y_val)
    Y_val = Y_val.reshape(Y_val.shape[0],Y_val.shape[1])
    Y_val = np.transpose(Y_val)
    print("shape y1 : ",Y1.shape)
    # Do the training loop
    # 1000 iterations
    mini_batch_size = 32
    m = X_train.shape[0]
    print("m ",m)
    costs = []
    seed = 0
    print_cost = True
    
    for epoch in range(100):
        epoch_cost = 0.                       # Defines a cost related to an epoch
        num_minibatches = int(m / mini_batch_size)
        # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y1 ,32, seed)
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={inputs: minibatch_X, labels: np.transpose(minibatch_Y) ,prob : 0.5})
            epoch_cost += minibatch_cost / num_minibatches
        # Print the cost every epoch
        if print_cost == True and epoch % 20 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            correct_prediction = tf.equal(tf.argmax(fc3,1), tf.argmax(labels,1))
            a_1 = sess.run(tf.count_nonzero(A1), feed_dict={inputs: X_train})
            print("A_1 :",a_1)
            # Calculate accuracy on the training set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            acc = sess.run(accuracy , feed_dict = {inputs: X_train, labels:np.transpose(Y1)})
            print("accuracy train set : ",acc)
            acc = sess.run(accuracy , feed_dict = {inputs: X_test, labels:np.transpose(Y_test)})
            print("accuracy test set : ",acc)

        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

    
    correct_prediction = tf.equal(tf.argmax(fc3,1), tf.argmax(labels,1))
    # Calculate accuracy on the training set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc= sess.run(accuracy , feed_dict = {inputs: X_val, labels:np.transpose(Y_val)})
    print("accuracy val set : ",acc)
    #saver.save(sess,'./lenet5-model')


'''
'''
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./lenet5_model")
    correct_prediction = tf.equal(tf.argmax(fc3), tf.argmax(labels))
    # Calculate accuracy on the training set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    Y1 = tf.one_hot(Y,10,axis = 1 )
    Y1 = sess.run(Y1)
    Y1 = Y1.reshape(Y1.shape[0],Y1.shape[1])
    print(X.shape)
    print(Y1.shape)
    print(correct_prediction.shape)
    fc3_res = sess.run(fc3 , feed_dict = {inputs: X, labels: Y1})
    print("fc3_res : ",fc3_res[1],Y1[1])
    accuracy = sess.run(accuracy , feed_dict = {inputs: X, labels: Y1})
    cost = sess.run(cost,feed_dict = {inputs: X, labels: Y1})
    print("cost : ", cost)
    print("accuracy : ",accuracy)
    print("number of correct prediction : ",correct_prediction.eval({inputs: X, labels: Y1}))
    #print("Train Accuracy:", accuracy.eval({inputs: X, labels: Y1}))
'''

    

