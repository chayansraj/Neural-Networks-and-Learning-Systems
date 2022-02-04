import numpy as np
from scipy import stats
#from utils import tanhprim

def solve_distance_tie(curr_distance, k):
    """Function to solve ties in distance for kNN implementation"""
    #n_ties = 1
    found_n_ties = False
    i = k+2 # We know that k+1 is a tie so start from k+2
    while(found_n_ties == False):
        if curr_distance[i] == curr_distance[k]:
            i += 1
        else:
            found_n_ties = True
    # Increase the k by number of points involved in the tie
    k = k + i
    return k

def majority_vote(curr_LTrain, k, Nclasses, classes):
    """Function to find the mode (majority vote) and resolve the issue when there are equally as many classes for kNN implementation"""
    nearest_label = curr_LTrain[0:k] # Get the nearest labels
    lst_Nclass = [0] * Nclasses #List of zeroes with length same as number of classes
    
    for i in range(0,len(nearest_label)): # Count the number of each class
        found_label = False
        j = 0
        while found_label == False:
            if nearest_label[i] == classes[j] :
                lst_Nclass[j] += 1
                found_label = True
            j += 1
    
    maximum = max(lst_Nclass) # Find the maxium number for each class (mode)
    maximum_index = np.argmax(lst_Nclass) # Index of mode
    
    all_maxes = [i for i in lst_Nclass if i == maximum] # See if other classes have the same value
    
    if len(all_maxes) > 1: # If there is not only one max
        if not k == len(curr_LTrain):
            label = majority_vote(curr_LTrain, k+1, Nclasses, classes) # Increment k by one and try again
        else: # If k is larger than number of trainig points we randomly sample a label
            label = np.random.choice(classes)
        return label
    else:
        label = maximum_index + 1
        return label
    

def kNN_CV(XTrain, LTrain, folds, hyperparam):
    """Function to perform k-fold crossvalidation for the number of neighbors in kNN"""
    n_samples = len(XTrain)
    X = XTrain
    Labels = LTrain
    # Reshuffle the data
    ind = np.random.permutation(range(len(Labels)))
    Labels, X = Labels[ind], X[ind]
    
    # Split the data: if n_samples/fold % 1 != 0 then the last subarray will get the rest
    X = np.array_split(X, folds)
    Labels = np.array_split(Labels, folds)
    
    accuracy = []
    
    for i in range(folds):
        #### Split the validation fold and the training fold
        Test_split = X[i]
        Test_labels = Labels[i]
        Train_split = np.delete(X, i, axis = 0)
        Train_split = np.concatenate(Train_split)
        Train_labels = np.delete(Labels, i, axis = 0)
        Train_labels = np.concatenate(Train_labels)
        #### Run the kNN algorithm
        LPred = kNN(Test_split, hyperparam, Train_split, Train_labels)
        #### Append the accuracy
        accuracy.append(sum(Test_labels == LPred) / len(Test_labels))
    # Return the mean accuracy
    return np.mean(accuracy)
        
        
def kNN(X, k, XTrain, LTrain):
    """ KNN: A fun
    Your implementation of the kNN algorithm
    
    Inputs:
            X      - Samples to be classified (matrix)
            k      - Number of neighbors (scalar)
            XTrain - Training samples (matrix)
            LTrain - Correct labels of each sample (vector)

    Output:
            LPred  - Predicted labels for each sample (vector)
    """
    
    
    classes = np.unique(LTrain) # The classes
    NClasses = classes.shape[0] # The number of classes
    classes = classes.tolist() # Make the classes a list object
    
    
    LPred = np.zeros((X.shape[0])) # An array of zeros; to store predicted labels

    for i in range(0,len(X)):
        curr_LTrain = LTrain
        curr_distance = np.array([]) # Array to store distances from the ith test point
        for j in range(0,len(XTrain)):
            curr_distance = np.append(curr_distance, np.sqrt(np.sum((X[i]-XTrain[j])**2))) # Euclidean distance (L2 norm)
        
        ind = np.lexsort((LTrain, curr_distance)) # Index for sort by distance, then by Label (if distances are equal class 1 will thus be chosen)
        ## Sort the two arrays with index
        # (this creates issues for ties. Will be resolved later)
        curr_LTrain = curr_LTrain[ind] # Sort the labels by the sorting index
        curr_distance = curr_distance[ind] # Sort the distance by the sorting index
        
        # If we see a tie based on distance we check how many points are involved in the tie and increase k to this number
        if curr_distance[k] == curr_distance[k+1]:
            if not k + 1 > len(XTrain):
                k = solve_distance_tie(curr_distance, k)
            else:
                k = len(XTrain)
        print(f"CurrDistance {i}: ")
        print(curr_distance)
        # Perform the majority vote
        if not k == len(XTrain):
            LPred[i] = majority_vote(curr_LTrain, k, NClasses,classes)
        else: # If k is > number of data points, randomly choose one of the labels (unlikely event)
            LPred[i] = np.random.choice(classes)

    return LPred


def runSingleLayer(X, W):
    """ RUNSINGLELAYER
    Performs one forward pass of the single layer network, i.e
    it takes the input data and calculates the output for each sample.

    Inputs:
            X - Samples to be classified (matrix)
            W - Weights of the neurons (matrix)

    Output:
            Y - Output for each sample and class (matrix)
            L - The resulting label of each sample (vector)
    """

    # Add your own code here
    Y = np.matmul(X,W)

    # Calculate labels
    L = np.argmax(Y, axis=1) + 1

    return Y, L


def trainSingleLayer(XTrain, DTrain, XTest, DTest, W0, numIterations, learningRate):
    """ TRAINSINGLELAYER
    Trains the single-layer network (Learning)
    
    Inputs:
            X* - Training/test samples (matrix)
            D* - Training/test desired output of net (matrix)
            W0 - Initial weights of the neurons (matrix)
            numIterations - Number of learning steps (scalar)
            learningRate  - The learning rate (scalar)
    Output:
            Wout - Weights after training (matrix)
            ErrTrain - The training error for each iteration (vector)
            ErrTest  - The test error for each iteration (vector)
    """

    # Initialize variables
    ErrTrain = np.zeros(numIterations+1)
    ErrTest  = np.zeros(numIterations+1)
    NTrain = XTrain.shape[0]
    NTest  = XTest.shape[0]
    Wout = W0

    # Calculate initial error
    YTrain, LTrain = runSingleLayer(XTrain, Wout)
    YTest, LTest  = runSingleLayer(XTest , Wout)
    ErrTrain[0] = ((YTrain - DTrain)**2).sum() / NTrain
    ErrTest[0]  = ((YTest  - DTest )**2).sum() / NTest

    for n in range(numIterations):
        # Add your own code here
        grad_w = 2/NTrain * np.matmul(XTrain.transpose(), (YTrain-DTrain))
        
        #print(f"Iteration {n}")

        # Take a learning step
        Wout = Wout - learningRate * grad_w

        # Evaluate errors
        YTrain, LTrain = runSingleLayer(XTrain, Wout)
        YTest, LTrain  = runSingleLayer(XTest , Wout)
        ErrTrain[n+1] = ((YTrain - DTrain) ** 2).sum() / NTrain
        ErrTest[n+1]  = ((YTest  - DTest ) ** 2).sum() / NTest

    return Wout, ErrTrain, ErrTest


def runMultiLayer(X, W, V):
    """ RUNMULTILAYER
    Calculates output and labels of the net
    
    Inputs:
            X - Data samples to be classified (matrix)
            W - Weights of the hidden neurons (matrix)
            V - Weights of the output neurons (matrix)

    Output:
            Y - Output for each sample and class (matrix)
            L - The resulting label of each sample (vector)
            H - Activation of hidden neurons (vector)
    """
    

    # Add your own code here
    S = np.matmul(X,W) # Calculate the weighted sum of input signals (hidden neuron)
    H = np.tanh(S)  # Calculate the activation of the hidden neurons (use hyperbolic tangent)
    Y = np.matmul(H,V)  # Calculate the weighted sum of the hidden neurons

    # Calculate labels
    L = Y.argmax(axis=1) + 1
    

    return Y, L, H


def trainMultiLayer(XTrain, DTrain, XTest, DTest, W0, V0, numIterations, learningRate):
    """ TRAINMULTILAYER
    Trains the multi-layer network (Learning)
    
    Inputs:
            X* - Training/test samples (matrix)
            D* - Training/test desired output of net (matrix)
            V0 - Initial weights of the output neurons (matrix)
            W0 - Initial weights of the hidden neurons (matrix)
            numIterations - Number of learning steps (scalar)
            learningRate  - The learning rate (scalar)

    Output:
            Wout - Weights after training (matrix)
            Vout - Weights after training (matrix)
            ErrTrain - The training error for each iteration (vector)
            ErrTest  - The test error for each iteration (vector)
    """

    # Initialize variables
    ErrTrain = np.zeros(numIterations+1)
    ErrTest  = np.zeros(numIterations+1)
    NTrain = XTrain.shape[0]
    NTest  = XTest.shape[0]
    NClasses = DTrain.shape[1]
    Wout = W0
    Vout = V0

    # Calculate initial error
    # YTrain = runMultiLayer(XTrain, W0, V0)
    YTrain, LTrain, HTrain = runMultiLayer(XTrain, Wout, Vout)
    YTest, LTest, HTest  = runMultiLayer(XTest , W0, V0)
    ErrTrain[0] = ((YTrain - DTrain)**2).sum() / (NTrain * NClasses)
    ErrTest[0]  = ((YTest  - DTest )**2).sum() / (NTest * NClasses)

    for n in range(numIterations):

        if not n % 1000:
            print(f'n : {n:d}')
            
        # Add your own code here
        grad_v =  2/NTrain * np.matmul(HTrain.transpose(), YTrain-DTrain)
        grad_w = 2/NTrain * np.matmul(XTrain.transpose(), (np.multiply(np.matmul(YTrain-DTrain, Vout.transpose()), (1-HTrain**2))))
        # 2 * XTrain(((XTrain%*%Wout - DTrain)%*%Vout.transpose()) (tanhprim(HTrain)))

        # Take a learning step
        Vout = Vout - learningRate * grad_v
        Wout = Wout - learningRate * grad_w

        # Evaluate errors
    #     YTrain = runMultiLayer(XTrain, Wout, Vout);
        YTrain, LTrain, HTrain = runMultiLayer(XTrain, Wout, Vout)
        YTest, LTest , HTest  = runMultiLayer(XTest , Wout, Vout)
        ErrTrain[1+n] = ((YTrain - DTrain)**2).sum() / (NTrain * NClasses)
        ErrTest[1+n]  = ((YTest  - DTest )**2).sum() / (NTest * NClasses)

    return Wout, Vout, ErrTrain, ErrTest