import math

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm

def load_data(file_path: str, label: str)->Tuple[np.ndarray, np.ndarray]:
    '''
    This function loads and parses text file separated by a ',' character and
    returns a data set as two arrays, an array of features, and an array of labels.
    Parameters
    ----------
    file_path : str
                path to the file containing the data set
    label : str
                A label of whether to grab training or testing data
    Returns
    -------
    features : ndarray
                2D array of shape (n,m) containing features for the data set
    labels : ndarray
                1D array of shape (n,) containing labels for the data set
    '''
    D = np.genfromtxt(file_path, delimiter=",")
    if label == "train":
        features = D[0:800, :-3]  # all columns but the last three
        labels = D[0:800, -3:]  # the last 3 columns
    else:
        features = D[801:1000, :-3]  # all columns but the last three
        labels = D[801:1000, -3:]  # the last 3 columns
    
    return features, labels

def initialize_network(layer_sizes : list, scale : float):
    """
    This function will inialize weights for an arbitrary neural network defined by
    the number of neurons in each layer, given by 'layer_sizes'
    Your weights should be initialized to be numpy arrays composed of random numbers
    taken from a gaussian distribution with mean=0 and std=1, then multiplied by scale
    Parameters
    ----------
    layer_sizes : np.ndarray
        An array of intergers that denote the number of neurons in each layer
        [100, 50, 20] would denote a network with 100 inputs, a hidden layer
        with 50 neurons and output of size 20 -- this would mean W_0 would have dimensions (100,50) for example
    scale : float
        The scale of our initial weights, our weights should mostly be in the range
        [-1,1] * scale

    Returns
    ---------
    init_params : dict
        A dictionary that maps labels for parameters to an array of those parameters'
        initial values
        You MUST have 'W0' map to the first weight matrix, 'W1' to the second, etc.
        Hint: "W" + str(1) is "W1"
    """
    # Initialize the parameter dictionary with weight matrices with random values
    # You need to use np.random.randn() to do this -- you can look up the API
    # This will give a number sampled from a normal distribution (a bell curve)
    init_params = {}
    for index in range(len(layer_sizes)-1):
        init_params['W' + str(index)] = scale * np.random.randn(layer_sizes[index],layer_sizes[index+1])
        init_params['b' + str(index)] = scale * np.random.randn(layer_sizes[index+1])

    return init_params

# Wieghts are matrices of size  input , output
# Biases : output size, 1
def sigma_forward(IN: np.ndarray):
    """
    performs the nonlinear function (sigmoid) on the given input and returns the result
    this is 1/(1 + e^IN)

    Parameters
    ----------
    IN: np.ndarray 
        The given input to some layer
    Returns
    ----------
    A: np.ndarray 
        sigma(IN), this is the A value of some layer
        This will need to be added to the cache too
    """

    ######################################

    # TODO Implement the sigma function 
    
    ######################################
    A = 1/(1 + np.exp(-IN))
    return A

def forward(params: dict, X: np.ndarray):
    """
    This function will perform the forward pass of your backprop algorithm
    It will calculate your networks prediction using your parameters and
    will keep a cache of all intermittent values (which you need for backprop)
    ** YOU MUST COMPLETE THE "sigmoid_forward()" method for this part **
    Parameters
    ----------
    params : dict
        A dictionary that maps the labels: 'W0', 'W1', etc to their respective
        weight matrices -- this is the current state of your params
    X : np.ndarray
        A 2D numpy array representing input where each row represents a feature vector
    
    Returns
    ---------
    prediction : np.ndarray
        A 1D numpy array holding the predictions of your network given input X
    cache : dict
        A dictionary that holds all of the intermittent values calculated during your
        forward pass of your network (the 'IN' and 'A'  of each layer), you must have the
        keys of this dictionary be of the form "AL" and "INL" where "AL" representes the input
        to the L-th layer of weights and "IN(L+1)" is the output after multiplying by weights in Layer L . 
        
        i.e "IN0" will be the key for exactly the input X and "A0" will be (as a special case) also X  
        generally "AL" will be sigma("INL")
    """
    # implement the forward pass of your network
    cache = {}
    # IN : W.T * X + b
    # A : sigmoid(IN)
    layerNumber = 0
    cache["A0"] = np.copy(X)
    cache["IN0"] = np.copy(X)
    while layerNumber < len(params) // 2:
        cache["IN" + str(layerNumber + 1)] = np.dot(cache["A" + str(layerNumber)], params["W" + str(layerNumber)]) + \
                                             params["b" + str(layerNumber)]
        if layerNumber < (len(params) // 2) - 1:
            cache["A" + str(layerNumber + 1)] = sigma_forward(cache["IN" + str(layerNumber + 1)])
        else:
            cache["A" + str(layerNumber + 1)] = cache["IN" + str(layerNumber + 1)]
        layerNumber += 1

    prediction = cache["A" + str(layerNumber)]


    return prediction, cache

def sigma_backward(A: np.ndarray):
    """
    calculates the derivative of your sigma function give the output of it
    Parameters
    ----------
    A: np.ndarray 
        sigmoid(IN), this is the A value (output of the sigma) of 
        some layer. This is all we need to find dsigma / dIN believe it or not
    Returns
    ----------
    dsigma: np.ndarray
        the derivative of sigma(IN) dIN -- this will use the A value
        it will also be very simple
    """
    # A = 1 / ( 1+ e^-IN ) => A *(1+e^-IN) = 1 => A + A*e^-IN = 1 => (1-A)/A = e^-IN
    ######################################
    # Implement the derivative of sigma
    ######################################
    dsigma = (1 - A) * A
    return dsigma


def backprop_and_loss(params: dict, prediction: np.ndarray, cache: dict, Y : np.ndarray):
    """
    This function will calculate the loss (LSE) of your predictions and the gradient
    of your network for a single iteration. To calculate the gradient you must
    use backpropogation through your network
    ** YOU MUST COMPLETE THE "sigma_backward()" method for this part **
    Parameters
    ----------
    params : dict
        A dictionary that maps the labels: 'W0', 'W1', etc to their respective
        weight as well as 'b0', 'b1', etc to the bias
        -- this is the current state of your params
    prediction : np.ndarray
        A 1D numpy array holding the predictions of your network given input X
    cache : dict
        A dictionary that holds all of the intermittent values calculated during your
        forward pass of your network (the 'IN' and 'A'  of each layer), you must have the
        keys of this dictionary be of the form "AL" and "INL" where "AL" representes the input
        to the L-th layer of weights and "IN(L+1)" is the output after multiplying by weights in Layer L . 
        
        i.e "IN0" will be the key for exactly the input X and "A0" will be (as a special case) also X 
        generally "AL" will be sigma("INL")
    Y : np.ndarray
        A 1D numpy array of the correct labels of our input X
    Returns
    ---------
    gradient : dict
        A dictionary that maps the labels: 'W0', 'W1', etc to the gradients of 
        the respective parameters (eg 'W0' -> gradient of first weight matrix)
    loss : float
        The MEAN (use np.mean) Squared Error loss given our predictions and true labels 'Y'. 
    
    """

    dLossWRTdPrediction = 2*(prediction - Y)
    dOut = dLossWRTdPrediction
    # Current Derivaitive calculated so far
    loss = np.mean((prediction - Y)**2)
    gradient = {}
    num_layers = len(params) // 2
    for index in reversed(range(num_layers)):
        dLoss_dwi = np.dot((cache["A"+str(index)]).T,dOut)
        gradient["W"+str(index)] = dLoss_dwi
        gradient["b" + str(index)] = np.mean(dOut)
        dLoss_dAi = dOut.dot(params["W" + str(index)].T) * sigma_backward(cache["A" + str(index)])
        dOut = dLoss_dAi
    # TODO calculate the gradients of each layer using backprop -- and calculate loss
    return gradient, loss

def gradient_descent(X : np.ndarray, Y : np.ndarray, initial_params : dict, lr : float, num_iterations : int)->Tuple[List[float], np.ndarray]:
    """
    This function runs gradient descent for a fixed number of iterations on the
    mean squared loss for a linear model parameterized by the weight vector w.
    This function returns a list of the losses for each iteration of gradient
    descent along with the final weight vector.
    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array representing input where each row represents a feature vector
    Y : np.ndarray
        A 1D numpy array where each element represents a label for MSE
    initial_params : dictionary
        A dictionary holding the initialization of all parameters of the model as np.ndarrays
        (e.g. key 'W0' maps to the first weight array of the neural net) 
    lr : float
        The step-size parameter to use with gradient descent.
    num_iterations : int
        The number of iterations of gradient descent to run.
    Returns
    -------
    losses : list
        A list of floats representing the loss from each iteration and the
        loss of the final weight vector
    final_params : dictionary 
        A dictionary holding all of the parameters after training as np.ndarrays
        (this should have the same mapping as initial_params, just with updated arrays) 
    """
    losses = []
    params = initial_params
    # Complete this function. It's the whole sh-bang (Gradient Descent)
    for n in tqdm(range(num_iterations)):  #tqdm will create a loading bar for your loop
        prediction,cache = forward(params, X)
        DerivativeParams,loss = backprop_and_loss(params, prediction,cache, Y)
        for param in params:
            params[param] -= lr * DerivativeParams[param]
        losses.append(loss)
    final_params = params
    return losses, final_params

def learning_curve(losses: list, names : list):
    """
    This function plots the learning curves for all gradient descent procedures in this homework.
    The plot is saved in the file learning_curve.png.
    Parameters
    ----------
    losses : list
        A list of arrays representing the losses for the gradient at each iteration for each run of gradient descent
    names : list
        A list of strings representing the names for each gradient descent method
    Returns
    -------
    nothing
    """
    for loss in losses:
        plt.plot(loss)
    plt.xscale("log")
    plt.ylim(0, 10000)
    plt.xlabel("Iterations")
    plt.ylabel("Squared Loss")
    plt.title("Gradient Descent")
    plt.legend(names)
    plt.savefig("learning_curve.png")
    plt.show()

def train_best_model(Train_X, Train_Y):
    """
    This function will train the model with the hyper parameters
    and layers that you have found to be best -- this model must get below 3
    MSE loss on our test data (which is not the test data you are given)
    """

    # TODO CHANGE THESE VALUES
    np.shape(Train_X)
    BEST_SCALE = 0.02         # You need
    BEST_LAYERS = [17,20, 3]           # to change
    BEST_ALPHA = 1e-6       # these
    BEST_NUM_ITERATIONS = 10000   # !
    # LR : e^-5
    # 1000 iterations : 1312.9642425305262
    # 10,000 Iterations: 263.5532959159481
    # 100000 Iterations: 301.5523573996149
    # Choosing 10000 Iterations
    # LR : e^-5 , 329.82868289845527,
    # LR : e^- 8,  1604.5941552175068
    # LR : e^-4 , 344.76074531959193
    # LR : e^-6 , 308.67533089876355 , Scale : 0.1,
    # LR : e^-6 , 306.96660619146996 , Scale : 0.01
    best_params = initialize_network(BEST_LAYERS, BEST_SCALE)
    best_losses, best_final_params = gradient_descent(Train_X, Train_Y, best_params, lr=BEST_ALPHA, num_iterations=BEST_NUM_ITERATIONS)

    return best_losses, best_final_params  

def hw1_data():
    D = np.genfromtxt('housing.csv', delimiter=",")
    features = D[:, :-1]  # all columns but the last one
    labels = D[:, -1]  # the last column
    lr = 0.001 # You can change this to test it out
    scale = 0.1 # ''
    n_iter = 100 # ''
    layers = [13,1] # the input size is 13 and output size is 1 so these must stay

    model = initialize_network(layers, scale=scale)
    losses, final_params = gradient_descent(features, labels, model, lr=lr, num_iterations=n_iter)
    learning_curve([losses], ['MLP on housing data'])  

def main():
    Train_X, Train_Y = load_data("StudentsPerformance.csv", "train")  # load the data set

    N = 10000 # N needs to equal 10,000 for your final plot. You can lower it to tune hyperparameters.

    init_params0 = initialize_network([17,3], scale=0.1) # initializes a sigle layer network (perceptron)
    losses0, final_params0 = gradient_descent(Train_X, Train_Y, init_params0, lr=1e-6, num_iterations=N)  

    init_params1 = initialize_network([17, 5, 3], scale=0.1)  # initializes a two layer network
    losses1, final_params1 = gradient_descent(Train_X, Train_Y, init_params1, lr=1e-6, num_iterations=N)  

    init_params2 = initialize_network([17, 7, 3, 3], scale=0.1)  # initializes a many layer network
    losses2, final_params2 = gradient_descent(Train_X, Train_Y, init_params2, lr=1e-6, num_iterations=N)   

    all_losses = [losses0, losses1, losses2]
    names = ["single layer", "two layer", "many layer"]
    print("final training loss values")
    for name, losses in zip(names, all_losses):
        print("{0:.<21}{1:>8.1f}".format(name, float(losses[-1])))

    learning_curve(all_losses, names)

    # TESTING 

    Test_X, Test_Y = load_data("StudentsPerformance.csv", "test")

    pred0, _ = forward(final_params0, Test_X)
    test_loss0 = np.mean(np.square(Test_Y[:, None]  - pred0))
    print("test loss of model 1:", test_loss0)

    pred1, _ = forward(final_params1, Test_X)
    test_loss1 = np.mean(np.square(Test_Y[:, None]  - pred1))
    print("test loss of model 2:", test_loss1)

    pred2, _ = forward(final_params2, Test_X)
    test_loss2 = np.mean(np.square(Test_Y[:, None]  - pred2))
    print("test loss of model 3:", test_loss2)

    # TODO choose the hyper parameters for your best model (change them in train_best_model() )
    print(Train_X.shape,"x.shape", Train_Y.shape,"y.shape")
    best_losses, best_params = train_best_model(Train_X, Train_Y) 
    best_pred, _ = forward(best_params, Test_X)

    best_loss = np.mean(np.square(Test_Y[:, None] - best_pred))
    print(Test_Y.shape,"Y dataset .shape",Test_X.shape,"X dataset.shape")
    print("test loss of your \"best\" model:", best_loss)
    plt.plot(best_losses)
    plt.xscale("log")
    plt.ylim(0, 10000)
    plt.xlabel("Iterations")
    plt.ylabel("Squared Loss")
    plt.title("Gradient Descent, Best Trained Model")
    plt.savefig("Best Model")
    plt.show()

if __name__ == "__main__":
    main()