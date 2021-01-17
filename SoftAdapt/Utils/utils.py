import numpy as np


def FD(loss_pts, args):

    if args["fd_order"] == 5:
        der = ((25/12) * loss_pts[4]) - ((4) * loss_pts[3]) + ((3) * loss_pts[2]) \
        - ((4/3) * loss_pts[1]) + ((1/4) * loss_pts[0])
        
    elif args["fd_order"] == 3:
        der = (-3/2) * loss_pts[0] + 2 * loss_pts[1] + (-1/2) * loss_pts[2]
    else:
        raise NotImplementedError("A finite difference order of {} is not implemented yet.".format(args.fd_order))
    
    return der




def set_hyper(alpha, args, loss_num):
    
    for i in range (0, loss_num): 
        args["alphas"][i] = alpha[i]
              
        
        
def get_5loss(new_loss, index, args):

    if args["global_count"] > 4 :
    
        if index == 1:
            args["loss2"] = np.hstack( (args["loss2"] , new_loss) );

            if args["global_count"] >= args["fd_order"]:
                args["loss2"] = args["loss2"][-args["fd_order"]:];

        elif index == 0:
            # to save the arrays in an orderly fashion
            args["loss1"] = np.hstack((args["loss1"], new_loss));
            if args["global_count"] >= args["fd_order"]:
                args["loss1"] = args["loss1"][-args["fd_order"]:];


        elif index == 2:
            args["loss3"] = np.hstack((args["loss3"], new_loss));
             # to save the arrays in an orderly fashion
            args.loss3.append(new_loss);

            if args.loss3.size > 5 :
                  args["loss3"] = args["loss3"][-5:];

        else :
             print("Wrong Index");

    
    else : 
        
        print("SoftAdapt idle --- less than 5 iters");
        args["global_count"] += 1;
 
 
    
 
def avg_calc(recent_loss1,args):
  
    if args["loss1_global_count"] > 4 :
        
        if args["loss1_avg"] == 0 : 
            
            args["loss1_avg"] = np.mean(args["loss2"]);
        
        args["loss1_avg"] = (args["loss1_avg"] * 5 + recent_loss1)/6;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
