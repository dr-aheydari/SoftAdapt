
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:52:38 2019

@author: aliheydari
@email: aliheydari@ucdavis.edu
@web: https://www.ali-heydari.com

"""
###################### ADAPTIVE LEARNING INTEGRATION ##########################
################ Adaptive Loss Functions WITHOUT AdaLearn #####################

import os
os.system("pip install easydict");
import numpy as np

version = "0.0.2"
backend = "PyTorch"


class SoftAdapt():
	
    def __init__(self, n, loss_tensor):
        self.n = n
        self.loss_tensor = loss_tensor
        self.Welcome_BEARD()


### Soft Adapt ###

    def SoftAdapt(self, beta, i):
     # numerator
    
	#      self.n = -1 * self.n;
      
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


### PlushAdapt ###

    def PlushAdapt(self, beta, i):

        #   n = -1 * n;

        if len(self.n) == 2 : 
            fe_x = np.zeros(2);
         
         # Normalize the slopes!!!!
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



### DOWNY SOFT ADAPT ###
         
    def DownyAdapt(self, beta, i):
        
    # numerator
        fe_x = np.zeros(2);
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

                            

    
    def alpha_assign(self, kappa, string):
    
       
        alpha = np.zeros(len(self.n));
        
        if string == "soft":
    
            for i in range(len(self.n)):
                    alpha[i] = SoftAdapt.SoftAdapt(self, kappa, i)
      
        if string == "plush":
       
            for i in range(len(self.n)):
                    alpha[i] = SoftAdapt.PlushAdapt(self, kappa, i)
                    
        if string == "downy":
    
            for i in range(len(self.n)):
                    alpha[i] = SoftAdapt.DownyAdapt(self, kappa, i)
                    
        
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
          print("SoftAdapt {} for {} imported succsessfuly".format(version, backend))
    

    

    
 
