
"""
@author: aliheydari
@email: aliheydari@ucdavis.edu
@web: https://www.ali-heydari.com

"""
################ SoftAdapt WITHOUT AdaLearn #####################

import os
import numpy as np

version = "0.0.3"



class SoftAdapt():
	
    def __init__(self, backend = "PyTorch"):
        """
        SoftAdapt:
        n -> number of loss components (for now, we have 2 and 3 but it can be easily extended)
        loss_tensor -> an initialized loss tensor which keeps track of the actual loss
        backend -> the backend ML library of the code, default is PyTorch
        """
#         self.n = slopes
#         self.loss_tensor = loss_tensor
        self.backend = backend
        self.Welcome_BEARD()


### Soft Adapt ###
    def SoftAdapt(self, beta, i):
     # numerator      
        if len(self.n) == 2 : 
     
            fe_x = np.zeros(2);
            fe_x[0] = self.loss_tensor[0].data.item() * np.exp(beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = self.loss_tensor[1].data.item() * np.exp(beta * (self.n[1] - np.max(self.n)));
            denom = fe_x[0] + fe_x[1];

        elif len(self.n) == 3 :
         
            fe_x = np.zeros(3);
            fe_x[0] = self.loss_tensor[0].data.item() * np.exp(beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = self.loss_tensor[1].data.item() * np.exp(beta * (self.n[1] - np.max(self.n)));
            fe_x[2] = self.loss_tensor[2].data.item() * np.exp(beta * (self.n[2] - np.max(self.n)));
            denom = fe_x[0] + fe_x[1] + fe_x[2];  
                                               
        else :
         	print("As of now, we only support 2 or 3 losses, please check input")
                                  
        return (fe_x[i]/ denom)


### Loss Weighted SofAdapt ###

    def LWAdapt(self, beta, i):
        if len(self.n) == 2 : 
            fe_x = np.zeros(2);
         
         # Normalize the slopes!
            self.n[0] = self.n[0] / (np.linalg.norm(self.n,1) + 1e-8);
            self.n[1] = self.n[1] / (np.linalg.norm(self.n,1) + 1e-8);
         
         # normalize the loss functions 
            denom2 = self.loss_tensor[0].data.item() + self.loss_tensor[1] 
    
            fe_x[0] = self.loss_tensor[0].data.item() / denom2;
            fe_x[1] = self.loss_tensor[1].data.item() / denom2;
        
            fe_x[0] = fe_x[0] * np.exp(beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = fe_x[1] * np.exp(beta * (self.n[1] - np.max(self.n)));
  
            denom = fe_x[0] + fe_x[1];                                      
         
            return (fe_x[i]/ denom)
         


        elif len(self.n) == 3 :
         
            fe_x = np.zeros(3);
         
         # Normalize the slopes
            self.n[0] = self.n[0] / (np.linalg.norm(self.n,1) + 1e-8);
            self.n[1] = self.n[1] / (np.linalg.norm(self.n,1) + 1e-8);
            self.n[2] = self.n[2] / (np.linalg.norm(self.n,1) + 1e-8);

         
         # Normalize the loss functions          
            denom2 = self.loss_tensor[0].data.item() + self.loss_tensor[1].data.item() + self.loss_tensor[3].data.item() 
    
            fe_x[0] = self.loss_tensor[0].data.item() / denom2;
            fe_x[1] = self.loss_tensor[1].data.item() / denom2;
            fe_x[2] = self.loss_tensor[2].data.item() / denom2;

        
            fe_x[0] = fe_x[0] * np.exp(beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = fe_x[1] * np.exp(beta * (self.n[1] - np.max(self.n)));
            fe_x[2] = fe_x[2] * np.exp(beta * (self.n[2] - np.max(self.n)));
  
            denom = fe_x[0] + fe_x[1] + fe_x[2] ;                                      
         
         
            return (fe_x[i]/ denom)
                                               
        else :
         
            print("As of now, we only support 2 or 3 losses, please check input")



### Normalized SoftAdapt ###
         
    def NormAdapt(self, beta, i):
        
    # numerator
        fe_x = np.zeros(2);
        # normalize the slopes
        self.n[0] = self.n[0] / (np.linalg.norm(self.n,1) + 1e-8);
        self.n[1] = self.n[1] / (np.linalg.norm(self.n,1) + 1e-8);
    
        denom2 = self.loss_tensor[0].data.item() + self.loss_tensor[1] 
    
        fe_x[0] = self.loss_tensor[0].data.item() / denom2;
        fe_x[1] = self.loss_tensor[1].data.item() / denom2;
    
        fe_x[0] = fe_x[0] * np.exp(beta * (self.n[0] - np.max(self.n)));
        fe_x[1] = fe_x[1] * np.exp(beta * (self.n[1] - np.max(self.n)));
  
        denom = fe_x[0] + fe_x[1];    

        if len(self.n) == 2 : 
     
            fe_x = np.zeros(2);
         
            # Normalize the slopes
            self.n[0] = self.n[0] / (np.linalg.norm(self.n,1) + 1e-8);
            self.n[1] = self.n[1] / (np.linalg.norm(self.n,1) + 1e-8);
         
            fe_x[0] = self.loss_tensor[0].data.item() * np.exp(beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = self.loss_tensor[1].data.item() * np.exp(beta * (self.n[1] - np.max(self.n)));
         
            denom = fe_x[0] + fe_x[1];
         
            return (fe_x[i]/ denom)


        elif len(self.n) == 3 :
         
            fe_x = np.zeros(3);
            self.n[0] = self.n[0] / np.linalg.norm(self.n,1);
            self.n[1] = self.n[1] / np.linalg.norm(self.n,1);
            self.n[2] = self.n[2] / np.linalg.norm(self.n,1); 
         
            fe_x[0] = self.loss_tensor[0].data.item() * np.exp(beta * (self.n[0] - np.max(self.n)));
            fe_x[1] = self.loss_tensor[1].data.item() * np.exp(beta * (self.n[1] - np.max(self.n)));
            fe_x[2] = self.loss_tensor[2].data.item() * np.exp(beta * (self.n[2] - np.max(self.n)));
            denom = fe_x[0] + fe_x[1] + fe_x[2]; 
            return (fe_x[i]/ denom)

                                               
        else :
            print("As of now, we only support 2 or 3 losses, please check input")

                            

    
    def alpha_assign(self, slopes, loss_tensor, beta=0.1, string="loss-weighted"):
        
        self.n = slopes
        self.loss_tensor = loss_tensor
        
        alpha = np.zeros(len(self.n));
        
        if string == "soft":
            for i in range(len(self.n)):
                    alpha[i] = SoftAdapt.SoftAdapt(self, beta, i)
      
        elif string == "normalized":
            for i in range(len(self.n)):
                    alpha[i] = SoftAdapt.NormAdapt(self, beta, i)
                    
        elif string == "loss-weighted":
            for i in range(len(self.n)):
                    alpha[i] = SoftAdapt.LWAdapt(self, beta, i)
                    
        return alpha 
    

    def Welcome_BEARD():
      
          print("\__________     __________/")
          print(" |         |-^-|         |")
          print(" |         |   |         |")
          print("  `._____.´     `._____.´")
          print("  \                     /")
          print("   \\\                 // ")
          print("    \\\    ////\\\\\\\   //")
          print("     \\\\\           /// ")
          print("       \\\\\\\\\\\\|////// ")
          print("         \\\\\\\\|//// ")
    
          print(" ")
          print(f"SoftAdapt {version} for {self.backend} imported succsessfuly")
    

    

    
 
