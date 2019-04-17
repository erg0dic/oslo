# -*- coding: utf-8 -*-
"""
Created on Sat Feb 02 21:21:02 2019

@author: Irtaza
"""
import numpy as np
import random as rn

class Site:
    """Define a site as a single column inside a pile of columns. Has attributes: threshold potential, height, postion"""
    
    def __init__(self, thresholds=[1,2], th_prob=[0.5,0.5]):
        self.height = 0
        self.thresholds = thresholds   # threshold potential paramenters
        self.probs = th_prob           # probability of thresholds
        self.thpotential = rn.randint(1,2)  # threshold potential using np.random.choice
    
    def increment(self):
         self.height += 1
         return self.height

    def decrement(self):
        self.height -= 1
        # decrementing always updates threshold
        self.thpotential = rn.randint(1,2)
       
    def get_height(self):
        return self.height
    
    def get_threshold(self):
        return self.thpotential
    
    def reset(self):
        self.height = 0
        self.thpotential = rn.randint(1,2)
        
#z = Site()
#
#print(z.increment()) 
#print(z.increment())   
#print(z.get_height())  
        

class Pile:
    """Define a pile and Oslo model parameters for an array of sites. Essentially an array of Site objects."""
    
    def __init__(self, size, iteration, thresholds=[1,2], th_prob=[0.5,0.5]):
        self.sites = [Site(thresholds, th_prob) for i in range(size)]
        self.iterations = iteration
        self.size = size                        # system size
        self.avalaunch_size = 0
        
#    def iterations(self):
#        return self.iteration
#    
#    def avalaunch_size(self):
#        return self.avalaunch_size
    
    def heights(self):
        return np.array([i.get_height() for i in self.sites])
    
    def thresholds(self):
        return np.array([i.get_threshold() for i in self.sites])
    
    def pile_height(self):
        "Total height of the pile defined as height of initial site 1."
        return self.sites[0].get_height()
    
    def drive(self):
        "Add a grain at site 1. As we're measuring time in terms of grains added, we decrement iterations."
        self.sites[0].increment()
#        self.iterations -= 1
        
#    def sites_to_relax(self):
#        "Compute sites that need to be relaxed."
#        z = self.heights()[:-1] - self.heights()[1:]  # z_i = h_i - h_{i+1} defined as the potential of single site
#        z = np.append(z, self.heights()[-1])  # as h_L+1 = 0 so z_L = h_L
#        return [i for i, site in enumerate(self.sites) if z[i] > site.get_threshold()]
#            
    def relax(self, i):
        "Relax an arbitrary site at index i."
        self.sites[i].decrement() 
        self.avalaunch_size += 1   # amounts to 1 avalaunch event
        if i == self.size-1: # if its the last site
            return    
        self.sites[i+1].increment() # grain topples over to the next site
    
    def event(self):
        "Define an event that comprises the oslo model running for a single grain added (drive)"
        
#        while True:
        self.avalaunch_size = 0     # set number of topples or avalauches to 0 before event start
        self.drive()   # drive step: add a grain            
        
        while True: # relaxation step
            z = self.heights()[:-1] - self.heights()[1:]  # z_i = h_i - h_{i+1} defined as the potential of single site
            z = np.append(z, self.heights()[-1])  # as h_L+1 = 0 so z_L = h_L            
            thresholds = self.thresholds()
            counter = 0
            for i in xrange(self.size): 
                if z[i] > thresholds[i]:
                    self.relax(i)
                    counter += 1
            if counter == 0:
                break                               # all sites relaxed or looped over


#           relax_sites = self.sites_to_relax()     # find sites to relax and then relax them
            
#            if not relax_sites:
#                break                       # all sites relaxed or looped over so return
#            
#            for i in relax_sites:
#                self.relax(i)   # relax sites at index i

#        if self.iterations == 0:
#            break
                              
    def startover(self):
        for site in self.sites:
            site.reset()

#        self.iterations = iteration
                
        
    
    
x = Pile(512, 1000)
counter = 0
for i in xrange(10000):
    x.event()
    counter += 1
    print(counter)

#print(x.heights())
##print(x.slopes())
#print(x.thresholds())
#print(x.sites_to_relax())