# SoftAdapt
The main page for SoftAdapt package. This work was developed during my internship at the Air Force Research Lab, and recently approved for public release. Different parts of the code may be released at different times, accompanied by comprehensive examples. Meanwhile, the  algorithm and the test cases can be found in our paper (SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks with Multi-Part Loss Functions)[https://arxiv.org/pdf/1912.12355.pdf] 


### Installing SoftAdapt

Once the repository is cloned, you can use `pip` to install the package (in the directory):
````
pip install -e .

````


### Part 1: SoftAdapt

Place the the <code>.py</code> files in the same folder as the code, the just import the name of the file (without .py) in the code.   




    
""" 
SOFTADAPT in various forms


Modified version of softmax with a little spice from beta

Beta is a hyperparameter that will either sharpen or dampen the peaks 

Default Beta is 0.1

Variations : 
    
*** SoftAdapt : A vanilla softmax with the value of the loss function 
        at each iteration multiplied by the exponent, i.e. 
        
        softAdapt(f,s) = f_i * e^(beta*s) / (sum f_je^(beta*s_j))
        where f is the loss value and s is the slope : 


*** PlushAdapt : The same idea as soft Adapt except the slopes and the loss
        function values are normalized at each iteration
        
        
        
*** DownyAdapt : Same as SoftAdapt except only the slopes are being normalized
        at each iteration       
        



Usage : 
    pass in a np vector n with a weight beta (if not sure what to use, then pass 1)
    returns softmax in the same dimensions as n

"""





  
    """
    Alpha assignment : a function that calls one of the variations of SoftAdapt
    and it will return the appropiate values for each alpha
    Usage : 
        
        Argument : A vector of slopes n
                   A constant value for the softmax called kappa (default1 1) 
                   A tensor of loss values at each iteration loss_tensor 
                       e.g. if your loss function is MSE + l1, then 
                       loss_tensor = [MSE, l1];
                   A string indicating which method you want to use
                   (default should be PlushAdapt)
                   """







############ FINITE DIFFERENCE #################

"""
loss Usage:
    
   pass in 5 points as a np array 
    outputs a forth order accurate first derivative approximation
    if more accurate slope approximation is needed, then more points would be 
     required
    
"""



"""
Set Hyper :
    sets the hyper parameters alpha0 through alpha n
    
    Usage : 
        
        takes in the vector Alpha, a dictionary of arguments (highly recommend
        a global dictionary) and number of loss functions
        outputs all the values of 
        
        
    Caution : 
        
        alpha -> the vector returned by alpha_assign() 
        alphas -> a global array that its entries are multiplied by the loss
        
"""


"""
Get 5 Loss :
    it stores the last 5 losses efficently and properly 
    
    Usage : 
        
       inputs :  the most current loss value -> new_loss 
                 a string that indicates which part of the loss is being stored
                     strings should be of the format "loss1", "loss2", "loss3" etc
                 
                 
                 
                 a (preferably global ) dictionary with the loss vectors
                     -> args
                 
                 
        output :  it sets the global arrays loss1 and loss2 and 
        their corresponding counters
        
"""





#################### avg_calc ############################
"""
average calculator : given the most recent loss it will calculate the average with resepct
to the average in a memory efficient way (I think)
input : the most recent loss (for each individual loss)
outputs : it changes the global variable args.loss1_avg (and for all other losses) and does
not return a value
"""





################## adapt_lr ##########################
"""
adaptive learning : it checks a very simple criterion and if it is true then it will set 
the learning rate to the max, and it will change a global flag and iter whcih then are used
to decay back down 
input : the most recent losses 
output : it will update the following global variables : 
        args.lr
        args.flag
        args.adapt_iter2
"""

   
################## lr_decay ##########################
"""
learning rate decay : it will increase or decrease the learning rate depending on the specific conditions
input : a dictionary of global variables args 
output : it will update the following global variables : 
        args.lr
        args.flag
        args.adapt_iter
"""


"""
Adaptive Learning Integration Function: Simply calling this function will 
Usage : 
    
    input: 
            which_SA : which variation of SoftAdapt you would want to use
                SoftAdapt ->  the original (weaker) version
                Plush SoftAdapt -> the more robust (defult) variation of SA
                Downey SoftAdapt -> A very smoothed out version of SA. Use 
                    with caution and for very specific cases
                    
            AdaLearn_bool: Whether you want to use AdaLearn or not
            
            






