import sys
sys.path.append("/Users/mac/Desktop/project/")
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import tensorflow as tf
import pickle
import math
import cv2
from scipy import ndimage



def extract_CrossValidationData(data ,PercentagePerDataSet = (60,20,20)):
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



def create_nnlayers(X_train , Y_train , conf_layers):
    '''
    Create neural network layers
    '''
    X = tf.placeholder(tf.float32, [X_train.shape[0], None], name="X")
    Y = tf.placeholder(tf.float32, [Y_train.shape[0], None], name="Y")
    W,B = create_hiddenlayers(conf_layers)

    return X,Y,W,B;

def create_hiddenlayers(conf_layers):
    '''
     Create weights according to the number of layers , and the configuration of neurons
    '''
    weights = []
    biases = []
    index = 1
    while index < len(conf_layers):
        weights.append(tf.get_variable('W'+str(index),[conf_layers[index],conf_layers[index-1]], initializer = tf.contrib.layers.xavier_initializer(seed=3)))
        biases.append(tf.get_variable('b'+str(index),[conf_layers[index],1], initializer = tf.zeros_initializer()))
        index+=1
    return weights , biases;

    

def encoding_onehot(labels , nbr_classes):
    '''
    Perform one hot encoding
    '''
    sess = tf.Session()
    onehot = sess.run(tf.one_hot(labels,nbr_classes , axis = 1 ))
    sess.close()
    return  onehot



def forward_propagation(X,W,B,training ,keep_prob = 0.7):

    index = 0
    #Activation values
    A = []
    
    for w in W :

        if index == 0 :
            z = tf.add(tf.matmul(w,X) , B[index])
            z_norm= tf.transpose(batch_norm_wrapper(z, training))
            a  = tf.nn.relu(z_norm)
            dropout_a = tf.nn.dropout(a , keep_prob)
            A.append(dropout_a)

        elif index == len(W) - 1 :
            z = tf.add(tf.matmul(w,A[index - 1]) , B[index])
            z_norm= tf.transpose(batch_norm_wrapper(z, training))
            A.append(z_norm)
        else :
            z = tf.add(tf.matmul(w,A[index - 1]) , B[index])
            z_norm= tf.transpose(batch_norm_wrapper(z, training))
            a  = tf.nn.relu(z_norm)
            dropout_a = tf.nn.dropout(a , keep_prob)
            A.append(dropout_a)
        
        index += 1
    return A[index - 1]

def compute_cost( Z , Y ):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost 


# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_norm_wrapper(inputs, is_training, decay = 0.999):



    print(inputs.get_shape()[1])
    scale = tf.Variable(tf.ones([inputs.get_shape()[0]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[0]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[0]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[0]]), trainable=False)
    batch_norm = tf.cond(tf.equal(is_training, tf.constant(True)), 
                         lambda: batchNormOnTrain(inputs,pop_mean, pop_var, beta, scale ,decay), 
                         lambda: tf.nn.batch_normalization(tf.transpose(inputs),pop_mean, pop_var, beta, scale , 0.001))        

    return batch_norm




def batchNormOnTrain(inputs,pop_mean , pop_var , beta, scale , decay ) :
    batch_mean, batch_var = tf.nn.moments(inputs,[1])
    train_mean = tf.assign(pop_mean,
                            pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var,
                         pop_var * decay + batch_var * (1 - decay))
    with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(tf.transpose(inputs),
            batch_mean, batch_var, beta, scale,0.001)


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
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def model ( X_data , Y_data , conf_hiddenLayer ,learning_rate = 0.0001 , mini_batch_size = 32 , num_epochs = 1500 , seed = 3 , print_cost = True):

        X_train , X_val , X_test = X_data
        Y_train , Y_val , Y_test = Y_data

        X_train = np.transpose(X_train)
        Y_train = np.transpose(Y_train)
        m = X_train.shape[1]
        conf_hiddenLayer.insert(0, X_train.shape[0])
        conf_hiddenLayer.append(Y_train.shape[0])
        print(conf_hiddenLayer)
        X,Y,W,B = create_nnlayers(X_train,Y_train, conf_hiddenLayer)
        isTraining = tf.placeholder(tf.bool)
        Z = forward_propagation(X,W,B , isTraining)
        cost = compute_cost(Z,Y)
        costs = []
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        '''
        with tf.Session() as sess:
        
            # Run the initialization
            sess.run(init)
        
             # Do the training loop
            for epoch in range(num_epochs):

                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / mini_batch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, mini_batch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y ,isTraining: True})
    
                
                    epoch_cost += minibatch_cost / num_minibatches

                
                epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

                saver.save(sess,"./dnn_model")
            
            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train ,isTraining: False}))
            print("Val Accuracy:", accuracy.eval({X: np.transpose(X_test), Y: np.transpose(Y_test) ,isTraining: False }))
            print("Test Accuracy:", accuracy.eval({X: np.transpose(X_val), Y: np.transpose(Y_val) ,isTraining: False}))
        '''
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, "./dnn_model")
            label_prediction= tf.argmax(Z)
            correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
            z_softmax = tf.nn.softmax(Z,0)
            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, isTraining: False}))
            print("Val Accuracy:", accuracy.eval({X: np.transpose(X_test), Y: np.transpose(Y_test), isTraining: False}))
            print("Test Accuracy:", accuracy.eval({X: np.transpose(X_val), Y: np.transpose(Y_val), isTraining: False}))
            with open('test.csv','r') as test_data:
                test_data = test_data.read()
                rows = test_data.split('\n')
                number_of_digits = len(rows[1:])
                print(number_of_digits)
                X_kg_test = np.ndarray(shape=(number_of_digits,784),dtype = np.float64)
                #Y = np.ndarray(shape=(number_of_digits,1),dtype = int)
                cpt=0
                for row in rows[1:]:
                
                    splitted_row = row.split(',')
                    #Y[cpt,:] = splitted_row[0]
                    X_kg_test[cpt,:] = splitted_row[0:]
                    cpt += 1
                X_kg_test = np.transpose(X_kg_test)
                X_kg_test = (X_kg_test - 255)/255
                predictions = label_prediction.eval({X:X_kg_test , isTraining : False})
                print(predictions)
                with  open('submission.csv','w') as submission_file:
                    i = 0
                    for prediction in predictions:
                        submission_file.write(str(i+1)+','+str(prediction)+'\n')
                        i += 1
                        

    