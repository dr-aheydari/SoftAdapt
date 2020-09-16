import numpy as np
from .SoftAdapt import SoftAdapt
from .Utils import utils

def make_args(*args):    
    
    args = {
        
        "alphas": np.zeros(2),
        "lr":0.001,
        "lr_max" : 0.01,
        "lr_min" : 0.0001,
        "loss1" : np.zeros(5),
        "loss2" : np.zeros(5),
        "global_count" : 0,
        "loss1_global_count" : 0,
        "loss2_global_count" : 0,
        "loss1_avg" : 0,
        "loss2_avg" : 0, 
        "flag": 'Go',
        "adapt_iter": 1,
        "adapt_iter2": 1,    
        "kappa": 1.5,
        "fd_order": 5,
        "num_epochs" : 30,
        "user_lazy": 'y'
    
     }    
    
    return args
    


def Adapt(recent_loss_tensor,which_SA="plush",AdaLearn_bool="False",beta=0.5\
        ,args = make_args()):
    
    # print all the parameters once at the very begining
    if args["global_count"] == 0 : 
      print("The Parameters of ALI : ");
      print("Soft Adapt Variation: {}".format(which_SA));
      print("Use AdaLearn: {}".format(AdaLearn_bool));      
      print("softmax Beta: {}".format(beta));
      print("learning rate : {}".format(args["lr"]))
      args["global_count"] += 1 ;
    
# def main(self):
    
    # get input : recent_loss_tensor,which_SA="plush",AdaLearn_bool="False"
    # create ALI object w/ inputs 
    #

    # if we want to use AdaLearn
    if AdaLearn_bool == True : 
        
        # calculate the running average in an efficient matter
        utils.avg_calc(recent_loss_tensor[0],recent_loss_tensor[1],args)
        
        # Set the Adaptive learning Rate
        # Decay the weight according to the performance 
        utils.lr_decay(args);
            
    
    if len(recent_loss_tensor) == 2:
        #store the last 5 losses for loss 1
        utils.get_5loss(recent_loss_tensor[0].data.item(),0,args);
        utils.get_5loss(recent_loss_tensor[1].data.item(),1,args);
                        
   
    elif len(recent_loss_tensor) == 3:
        utils.get_5loss(recent_loss_tensor[0].data.item(),0,args);
        utils.get_5loss(recent_loss_tensor[1].data.item(),1,args);
        utils.get_5loss(recent_loss_tensor[2].data.item(),2,args);
    
    else : 
        
        print("support for this coming soon");
    
    
    if args["global_count"] > 4 :
        
         if len(recent_loss_tensor) == 2 :
             slopes = np.zeros(2);
             slopes[0] = utils.FD(args["loss1"],args);
             slopes[1] = utils.FD(args["loss2"],args);
             
             utils.set_hyper(SoftAdapt.alpha_assign(slopes,beta,recent_loss_tensor,which_SA)\
                       ,args,len(recent_loss_tensor));
             
        
         elif len(recent_loss_tensor) == 3 :
             slopes = np.zeros(3);
             slopes[0] = utils.FD(args["loss1"],args);
             slopes[1] = utils.FD(args["loss2"],args);
             slopes[2] = utils.FD(args["loss3"],args);
                
             utils.set_hyper(SoftAdapt.alpha_assign(slopes,beta,recent_loss_tensor,which_SA)\
                       ,args,len(recent_loss_tensor));
        
         else:
             
             print("Implementation coming soon : ) ")
                
        
# to see if we need to return alpha or not for the user
    #if hasattr(args, 'user_lazy'):
    if 1 < 2 :   
       alpha_return = np.zeros(len(recent_loss_tensor))
      
      
       # if user did not make the global dictionary
       #if args['user_lazy'] == 'y' : 
       if 1 < 2 : 
           # return the alphas
           alpha_return = args['alphas'];

       return alpha_return
    
