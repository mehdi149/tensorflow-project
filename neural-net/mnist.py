import tf_dnn
import numpy as np

if __name__ == "__main__":
    
    labels = [0,1,2,3,4,5,6,7,8,9]
    with open('train.csv','r') as training_file:
        index=0
        training_data = training_file.read()
        rows = training_data.split('\n')
        number_of_digits = len(rows[1:])
        print(number_of_digits)
        X = np.ndarray(shape=(number_of_digits,784),dtype = np.float64)
        Y = np.ndarray(shape=(number_of_digits,1),dtype = int)
	    
        for row in rows[1:]:
            splitted_row = row.split(',')
            Y[index,:] = splitted_row[0]
            X[index,:] = splitted_row[1:]
            index += 1

    index = 0
    for label in Y:
        Y[index,0] = labels.index(Y[index,0])
        index+=1

   
    Y = tf_dnn.encoding_onehot(Y,10)
    Y = Y.reshape(Y.shape[0],Y.shape[1])
    X = (X - 255)/255

    data = {'X':X ,'Y':Y}

    splited_dataset = tf_dnn.extract_CrossValidationData(data)
    X_tuple = (splited_dataset['X_train'] ,splited_dataset['X_val'] ,splited_dataset['X_test'])
    Y_tuple = (splited_dataset['Y_train'] ,splited_dataset['Y_val'] ,splited_dataset['Y_test'])
    tf_dnn.model(X_tuple,Y_tuple ,[100,75,25])
