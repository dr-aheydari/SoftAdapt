import numpy as np
from .SoftAdapt import SoftAdapt
from .Utils import utils

def make_args(*args, alphas=2, kappa=1.5, fd_order=5, num_epochs=100):    
    
    args = {
        
        "alphas": np.array((0.5,0.5)),
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
        "kappa": kappa,
        "fd_order": fd_order,
        "num_epochs" : num_epochs,
        "user_lazy": 'y'
    
     }    
    
    return args
    


def Adapt(recent_loss_tensor, which_SA="loss-weighted", beta=0.5, args = make_args()):
    
    SA = SoftAdapt(backend = "PyTorch");
    # print all the parameters once at the very begining
    if args["global_count"] == 0 : 
        print("The Parameters of SoftAdapt : ");
        print(f"    -> Soft Adapt Variation: {which_SA}"); 
        print(f"    -> softmax Beta: {beta}");
        args["global_count"] += 1 ;
        #create a SoftAdapt object
    
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
            utils.set_hyper(SA.alpha_assign(slopes, recent_loss_tensor, beta=beta, string=which_SA),
                            args,
                            len(recent_loss_tensor)
                            );
             
        
         elif len(recent_loss_tensor) == 3 :
            slopes = np.zeros(3);
            slopes[0] = utils.FD(args["loss1"],args);
            slopes[1] = utils.FD(args["loss2"],args);
            slopes[2] = utils.FD(args["loss3"],args);
                
            utils.set_hyper(SA.alpha_assign(slopes, recent_loss_tensor, beta=beta, string=which_SA),
                            args,
                            len(recent_loss_tensor)
                            );
        
        # more than 3 loss components
         else:
             print("Implementation coming soon : ) ")
     

              
    alpha_return = args['alphas']
    return alpha_return
                
        
# # to see if we need to return alpha or not for the user
#     if hasattr(args, 'user_lazy'):
#        alpha_return = np.zeros(len(recent_loss_tensor))
      
#        # if user did not make the global dictionary
#        if args['user_lazy'] == 'y' : 
#            # return the alphas
#            ;

    
