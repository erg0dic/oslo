# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import random as rn
#### Oslo d = 1 routine

def maxitime(system_size):
    "worse case max system size happens for threshold which is 2 and we can just compute the total number of grains in the system for maxittime"
    sum = 0
    L = system_size
    for i in range(1, L+1):
        sum += 2*i
    return sum


def oslo(system_size, itime): # faulty!!!!!!!
    "Implements the oslo model and returns steady state heights"

    ################### initialization #############################
    
    L = system_size + 1     #system_size
    
    sites = np.zeros(L)   # initialize system heights
    z = sites
    
    heights = np.zeros(L)  # store heights
    h  = heights
    
    thresholds = np.random.randint(1, 3, size = L)  # initialize threshold slopes
    z_th = thresholds
    
    ################## iteration step #############################
    
    counter2 = 0
    
    
    while True:
        counter = 0
        for i in range(len(sites)):
            if i == 0:   # drive
                z[i] += 1
                h[i] += 1
                
            if z[i] > z_th[i]:  # relaxation
                
                if i == 0:
                    z[i] -= 2
                    z[i+1] += 1
                    
                    h[i] -= 1           # update heights
                    h[i+1] += 1
                    
                    z_th[i] = rn.randint(1,2) # change threshold upon relaxation 
                    
                elif i != 0 and (i != len(sites)-1):
                    z[i] -= 2
                    z[i+1] += 1
                    z[i-1] += 1
                    
                    
                    h[i] -= 1           # update heights
                    h[i+1] += 1                
                    
                    z_th[i] = rn.randint(1,2) # change threshold upon relaxation
                    
                elif (i == len(sites)-1):
                   z[i] -= 1
                   z[i-1] += 1
                   
                   h[i] -= 1           # update heights  (different: grain leaves the system)              
                   
                   z_th[i] = rn.randint(1,2) # change threshold upon relaxation
                   
            if z[len(sites) - 1] != 0:    # don't break the loop at the beginning
                
                if z[i] <= z_th[i]:   # exit condition linked with the fact that system in steady state
                    counter += 1
        counter2 += 1
        
        if counter == len(sites):   # store iteration time taken to reach steady state
            itsteady = counter2
        
        
        if counter2 == itime:       # implement exit condition if reached for all sites eg. for how many iterations I want it to run
            break
    
    return h, itsteady

L = 32
iterations = 1000
#site1h = np.zeros(iterations)     # heights per time step 
#
#for i in range(iterations):
#    site1h[i] += oslo(L)[0]
#    
#print(np.average(site1h))


def oslost(system_size, itime, thresholds_z=[1,2], probs=[0.5,0.5]):
    "Implements the oslo model and returns steady state height of the first site averaged over itime iterations"

    ################### initialization #############################
    
    L = system_size     #system_size
    
    sites = np.zeros(L)   # initialize system heights
    z = sites
    
    heights = np.zeros(L)  # store heights
    h  = heights
    
    thresholds = np.random.randint(thresholds_z[0], thresholds_z[1]+1, size=L) #np.random.choice([1,2], size=L, p = [0.5, 0.5]) # initialize threshold slopes
    z_th = thresholds
    
    ################## iteration step #############################
    
    counter2 = 0    # initialize counter that counts number of iterations the oslo model runs for
    
    itsteady = 0    # initialize steady state iteration number
    
    steady = 0    # binary to make the algo know steady state has been reached
    
    while True:
        counter5 = 0
#        z[0] += 1   # drive
        h[0] += 1
        
        while True:   # relaxation
#            counter = 0
            i = 0
            counter = 0
            while i < len(sites):
                if i != len(sites)-1:
                    z[i] = h[i] - h[i+1]
                if i == len(sites)-1:
                    z[i] = h[i]
                    
                if z[i] > z_th[i]:
                    counter += 1
#                        z[i] -= 2
#                        z[i+1] += 1
                    
                    h[i] -= 1           # update heights
 #                       z[i] = h[i] - h[i+1]
                    z_th[i] = rn.randint(thresholds_z[0], thresholds_z[1]) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation 
                    
                    counter5 += 1
                    if i == len(sites)-1:
                        break
                    h[i+1] += 1   
#                    elif i != 0 and (i != len(sites)-1):
##                        z[i] -= 2
##                        z[i+1] += 1
##                        z[i-1] += 1
#                        
#                        
#                        h[i] -= 1           # update heights
#                        h[i+1] += 1                
#                        
#                        z_th[i] = rn.randint(1,2) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation
#                        
#                        counter5 += 1
#                        
#                    elif (i == len(sites)-1):
##                       z[i] -= 1
##                       z[i-1] += 1
#                       
#                       h[i] -= 1           # update heights  (different: grain leaves the system)              
#                       
#                       z_th[i] = rn.randint(1,2) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation
#                       
#                       counter5 += 1
                i += 1
                #if h[0] != 0:    # don't break the loop at the beginning
#                if z[i] <= z_th[i]:   
#                    counter += 1
#            counter = 0            
#            for i in range(0, len(sites)): # exit condition linked with the fact that system is fully relaxed
#                if z_th[i] >= z[i]:
#                    counter += 1
            if counter == 0:
                break
#            if np.all((z_th - z) > np.zeros(len(sites))) == True:
#                break
#            print(counter2)
        counter2 += 1

        #print(float(counter2)/maxitime(L))
        
#        if counter2 == maxitime(L) and steady == 0:   # store iteration time taken to reach steady state
#            itsteady = counter2   # roughly steady
#            ## initialize height recording
#            site1h = np.zeros(itime-counter2+1)     # heights per time step
#            site1h[counter2-itsteady] = h[0]
#            steady = 1   # flip switch so this condition is no longer triggered
#
#        if counter2 > itsteady and itsteady != 0:
#            site1h[counter2-itsteady] = h[0]
#            #print(site1h)
        
        if counter2 == itime:       # implement exit condition if reached for all sites eg. for how many iterations I want it to run
            break
        

    
    return heights 


def oslo_pile_heights(system_size, itime, thresholds_z=[1,2], probs=[0.5,0.5]):
    "Implements the oslo model and returns steady state height of the first site averaged over itime iterations"

    ################### initialization #############################
    
    L = system_size     #system_size
    
    sites = np.zeros(L)   # initialize system heights
    z = sites
    
    heights = np.zeros(L)  # store heights
    h  = heights
    
    thresholds = np.random.randint(thresholds_z[0], thresholds_z[1]+1, size=L) #np.random.choice([1,2], size=L, p = [0.5, 0.5]) # initialize threshold slopes
    z_th = thresholds
    
    pile_heights = np.zeros(itime)   # defined as the height of the first site h[0] over time
    
    
    ################## iteration step #############################
    
    added_grains = 0    # initialize counter that counts number of iterations the oslo model runs for
    

    
#    steady_first_time = False    # binary to make the algo know steady state has been reached
    
    while True:
        ava_size = 0
#        z[0] += 1   # drive
        h[0] += 1
        
        while True:   # relaxation
#            counter = 0
            i = 0
            counter = 0
            while i < len(sites):
                if i != len(sites)-1:
                    z[i] = h[i] - h[i+1]
                if i == len(sites)-1:
                    z[i] = h[i]
                    
                if z[i] > z_th[i]:
                    counter += 1
#                        z[i] -= 2
#                        z[i+1] += 1
                    
                    h[i] -= 1           # update heights
 #                       z[i] = h[i] - h[i+1]
                    z_th[i] = rn.randint(thresholds_z[0], thresholds_z[1]) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation 
                    
                    ava_size += 1
                    if i == len(sites)-1:
                        break
                    h[i+1] += 1   
#                    elif i != 0 and (i != len(sites)-1):
##                        z[i] -= 2
##                        z[i+1] += 1
##                        z[i-1] += 1
#                        
#                        
#                        h[i] -= 1           # update heights
#                        h[i+1] += 1                
#                        
#                        z_th[i] = rn.randint(1,2) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation
#                        
#                        counter5 += 1
#                        
#                    elif (i == len(sites)-1):
##                       z[i] -= 1
##                       z[i-1] += 1
#                       
#                       h[i] -= 1           # update heights  (different: grain leaves the system)              
#                       
#                       z_th[i] = rn.randint(1,2) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation
#                       
#                       counter5 += 1
                    
                i += 1
                #if h[0] != 0:    # don't break the loop at the beginning
#                if z[i] <= z_th[i]:   
#                    counter += 1
#            counter = 0            
#            for i in range(0, len(sites)): # exit condition linked with the fact that system is fully relaxed
#                if z_th[i] >= z[i]:
#                    counter += 1
            if counter == 0:
                break
#            if np.all((z_th - z) > np.zeros(len(sites))) == True:
#                break
#ax            print(added_grains)
        
        pile_heights[added_grains] = h[0]
        added_grains += 1
        

        

        #print(float(counter2)/maxitime(L))
        
#        if counter2 == maxitime(L) and steady == 0:   # store iteration time taken to reach steady state
#            itsteady = counter2   # roughly steady
#            ## initialize height recording
#            site1h = np.zeros(itime-counter2+1)     # heights per time step
#            site1h[counter2-itsteady] = h[0]
#            steady = 1   # flip switch so this condition is no longer triggered
#
#        if counter2 > itsteady and itsteady != 0:
#            site1h[counter2-itsteady] = h[0]
#            #print(site1h)
        
        if added_grains == itime:       # implement exit condition if reached for all sites eg. for how many iterations I want it to run
            break
        

    
    return pile_heights 


#print(oslo_pile_heights(16, 2*maxitime(16)))


########################################### code for task 2a

#plt.figure(1)
#for i in [4, 8, 16, 32, 64, 128, 256]:
#
#    if i > 64:
#        x = oslo_pile_heights(i, int(1.5*maxitime(256)))
#        plt.plot(range(int(1.5*maxitime(256))), x, '-', label = 'sites L = {0}; Steady average height = {1}'.format(i, round(np.average(x[maxitime(i):]), 1)))
#    if i <= 64:
#        x = oslo_pile_heights(i, int(1.5*maxitime(256)))
#        plt.plot(range(int(1.5*maxitime(256))), x, '-', label = 'sites L = {0}; Steady average height = {1}'.format(i, round(np.average(x[maxitime(i):]), 1)))
#plt.xlabel('iterations time (in terms of added grains)')
#plt.ylabel('Pile heights (of site 1)')
#plt.legend()



def rolling_average_ph(size, itime, M):
    "Rolling average of pile heights of the oslo model. M many times computed."
    global_av = np.zeros(itime)
    for m in xrange(M):
        pile_h = oslo_pile_heights(size, itime)
        global_av += pile_h
    global_av /= float(M)
    return global_av

#print(rolling_average_ph(16, 2*maxitime(16), 100))

############################################## task 2c
    
#plt.figure(1)
#for i in [4, 8, 16, 32, 64, 128, 256]:
#
#    if i > 64:
#        time = np.arange(int(1.5*maxitime(256)))
#        x = rolling_average_ph(i, int(1.5*maxitime(256)), 50)/float(i)
#        plt.loglog(time/float(i*i), x, '-', label = 'sites L = {0}'.format(i))
#    if i <= 64:
#        time = np.arange(int(1.5*maxitime(256)))
#        x = rolling_average_ph(i, int(1.5*maxitime(256)), 50)/float(i)
#        plt.loglog(time/float(i*i), x, '-', label = 'sites L = {0}'.format(i))
#plt.xlabel(r'$\frac{\tilde{t}}{L}$')
#plt.ylabel(r'$\frac{\tilde{h}}{L^2}$')
#plt.legend()
   
def oslo_crosst(system_size, itime, thresholds_z=[1,2], probs=[0.5,0.5]):
    "returns cross time."

    ################### initialization #############################
    
    L = system_size     #system_size
    
    sites = np.zeros(L)   # initialize system heights
    z = sites
    
    heights = np.zeros(L)  # store heights
    h  = heights
    
    thresholds = np.random.randint(thresholds_z[0], thresholds_z[1]+1, size=L) #np.random.choice([1,2], size=L, p = [0.5, 0.5]) # initialize threshold slopes
    z_th = thresholds
    
    steady_first_time = False                  # binary condition pile to become get steady for the first time

    
    
    ################## iteration step #############################
    
    added_grains = 0    # initialize counter that counts number of iterations the oslo model runs for
    
    
    while True:
        counter5 = 0
        h[0] += 1
        
        while True:   # relaxation
#            counter = 0
            i = 0
            counter = 0
            while i < len(sites):
                if i != len(sites)-1:
                    z[i] = h[i] - h[i+1]
                if i == len(sites)-1:
                    z[i] = h[i]
                    
                if z[i] > z_th[i]:
                    counter += 1            
                    
                    h[i] -= 1           # update heights

                    z_th[i] = rn.randint(thresholds_z[0], thresholds_z[1]) # change threshold upon relaxation 
                    
                    counter5 += 1
                    if i == len(sites)-1:
#                        added_grains += 1
                            # grain leaves pile for the first time                       
#                        return added_grains
                        if steady_first_time == False:
                            print(heights)
#                            print(z_th)
                            steady_first_time = True
                            return added_grains
                        break
                    h[i+1] += 1   
                i+=1
            if counter == 0:
                break
            
            
#            print(added_grains)
        
        added_grains += 1
        
        
        

        #print(float(counter2)/maxitime(L))
        
#        if counter2 == maxitime(L) and steady == 0:   # store iteration time taken to reach steady state
#            itsteady = counter2   # roughly steady
#            ## initialize height recording
#            site1h = np.zeros(itime-counter2+1)     # heights per time step
#            site1h[counter2-itsteady] = h[0]
#            steady = 1   # flip switch so this condition is no longer triggered
#
#        if counter2 > itsteady and itsteady != 0:
#            site1h[counter2-itsteady] = h[0]
#            #print(site1h)
        
        if added_grains == itime:       # implement exit condition if reached for all sites eg. for how many iterations I want it to run
            break
        
    return heights

def oslo_pile_heights2(system_size, itime, thresholds_z=[1,2], probs=[0.5,0.5]):
    "Implements the oslo model and returns steady state height of the first site averaged over itime iterations"

    ################### initialization #############################
    
    L = system_size     #system_size
    
    sites = np.zeros(L)   # initialize system heights
    z = sites
    
    heights = np.zeros(L)  # store heights
    h  = heights
    
    thresholds = np.random.randint(thresholds_z[0], thresholds_z[1]+1, size=L) #np.random.choice([1,2], size=L, p = [0.5, 0.5]) # initialize threshold slopes
    z_th = thresholds
    
    pile_heights = np.zeros(itime)   # defined as the height of the first site h[0] over time
    
    
    ################## iteration step #############################
    
    added_grains = 0    # initialize counter that counts number of iterations the oslo model runs for
    
    itsteady = 0    # initialize steady state iteration number
    ava_arr = []
    freq = []
    
    steady_first_time = False    # binary to make the algo know steady state has been reached
    
    while True:
        ava_size = 0
#        z[0] += 1   # drive
        h[0] += 1
        
        while True:   # relaxation
#            counter = 0
            i = 0
            counter = 0
            while i < len(sites):
                if i != len(sites)-1:
                    z[i] = h[i] - h[i+1]
                if i == len(sites)-1:
                    z[i] = h[i]
                    
                if z[i] > z_th[i]:
                    counter += 1
#                        z[i] -= 2
#                        z[i+1] += 1
                    
                    h[i] -= 1           # update heights
 #                       z[i] = h[i] - h[i+1]
                    z_th[i] = rn.randint(thresholds_z[0], thresholds_z[1]) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation 
                    
                    ava_size += 1
                    if i == len(sites)-1:
                        if steady_first_time == False:
                            itsteady = added_grains+1
                            steady_first_time = True
                        break
                    h[i+1] += 1   
#                    elif i != 0 and (i != len(sites)-1):
##                        z[i] -= 2
##                        z[i+1] += 1
##                        z[i-1] += 1
#                        
#                        
#                        h[i] -= 1           # update heights
#                        h[i+1] += 1                
#                        
#                        z_th[i] = rn.randint(1,2) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation
#                        
#                        counter5 += 1
#                        
#                    elif (i == len(sites)-1):
##                       z[i] -= 1
##                       z[i-1] += 1
#                       
#                       h[i] -= 1           # update heights  (different: grain leaves the system)              
#                       
#                       z_th[i] = rn.randint(1,2) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation
#                       
#                       counter5 += 1
                    
                i += 1
                #if h[0] != 0:    # don't break the loop at the beginning
#                if z[i] <= z_th[i]:   
#                    counter += 1
#            counter = 0            
#            for i in range(0, len(sites)): # exit condition linked with the fact that system is fully relaxed
#                if z_th[i] >= z[i]:
#                    counter += 1
            if counter == 0:
                break
#            if np.all((z_th - z) > np.zeros(len(sites))) == True:
#                break
            print(added_grains)
        
        pile_heights[added_grains] = h[0]
        added_grains += 1
        
        if steady_first_time == True:
            if ava_size not in ava_arr:
                ava_arr.append(ava_size)
                freq.append(1)
            if ava_size in ava_arr:
                freq[ava_arr.index(ava_size)] += 1
        

        #print(float(counter2)/maxitime(L))
        
#        if counter2 == maxitime(L) and steady == 0:   # store iteration time taken to reach steady state
#            itsteady = counter2   # roughly steady
#            ## initialize height recording
#            site1h = np.zeros(itime-counter2+1)     # heights per time step
#            site1h[counter2-itsteady] = h[0]
#            steady = 1   # flip switch so this condition is no longer triggered
#
#        if counter2 > itsteady and itsteady != 0:
#            site1h[counter2-itsteady] = h[0]
#            #print(site1h)
        
        if added_grains == itime:       # implement exit condition if reached for all sites eg. for how many iterations I want it to run
            break
        

    
    return [pile_heights, np.stack((ava_arr, freq), axis=-1), itsteady]

       
#print(oslo_pile_heights(512, 2*maxitime(512)))

#################### task 2d

#sum = 0
#counter=0
#for i in range(10):
#    sum += oslo_crosst(128, maxitime(128))
#    counter+=1
#    print(counter)
#sum /= float(10)
#sum
 
import pickle
#    
#counter = 0
#for size in [1024]:
##    
#    with open(('system_size_take3_%s.txt' % size), 'wb') as outfile:
#        # I'm writing a header here for the sake of readability
#        # Any line starting with "#" will be ignored by numpy.loadtxt
#        
#        x = oslo_pile_heights2(size, maxitime(1024)+3000000)
#        # Iterating through a ndimensional array produces slices along
#        # the last axis. This is equivalent to data[i,:,:] in this case
#
#        pickle.dump(x, outfile)
#
#    counter+= 1
#    print(counter)
#        for data_slice in data_set:
#    
#            # The formatting string indicates that I'm writing out
#            # fmt type to make sure the right decimal accuracy is preserved
#            np.savetxt(outfile, data_slice, fmt='%5.12g') #'%-7.2f' '%.4e'
#    
#            # Writing out a break to indicate different slices...
#            outfile.write('# New vector with column format (xa, ya, za, xb, yb, zb, vne) \n')

#counter = 0
#for size in [4, 8, 16, 32, 64, 128, 256, 512]:
#
#    with open(('rolled_pile_heights%s.txt' % size), 'wb') as outfile:
#        if size < 128:  
#            x = rolling_average_ph(size, int(1.5*maxitime(128)), 50)
#        elif size > 128 and size < 512:
#            x = rolling_average_ph(size, int(1.5*maxitime(256)), 50)
#        elif size == 512:
#            x = rolling_average_ph(size, int(1.5*maxitime(512)), 10)
#        pickle.dump(x, outfile)
#
#    counter+= 1
#    print(counter)            
                

def oslo_crosst_revised(system_size, itime, thresholds_z=[1,2], probs=[0.5,0.5]):
    "Implements the oslo model and returns steady state height of the first site averaged over itime iterations"

    ################### initialization #############################
    
    L = system_size     #system_size
    
    sites = np.zeros(L)   # initialize system heights
    z = sites
    
    heights = np.zeros(L)  # store heights
    h  = heights
    
    thresholds = np.random.randint(thresholds_z[0], thresholds_z[1]+1, size=L) #np.random.choice([1,2], size=L, p = [0.5, 0.5]) # initialize threshold slopes
    z_th = thresholds
    
    pile_heights = np.zeros(itime)   # defined as the height of the first site h[0] over time
    
    
    ################## iteration step #############################
    
    added_grains = 0    # initialize counter that counts number of iterations the oslo model runs for
    
    itsteady = 0    # initialize steady state iteration number
    ava_arr = np.array([])
    
    steady_first_time = False    # binary to make the algo know steady state has been reached
    
    while True:
        ava_size = 0
#        z[0] += 1   # drive
        h[0] += 1
        
        while True:   # relaxation
#            counter = 0
            i = 0
            counter = 0
            while i < len(sites):
                if i != len(sites)-1:
                    z[i] = h[i] - h[i+1]
                if i == len(sites)-1:
                    z[i] = h[i]
                    
                if z[i] > z_th[i]:
                    counter += 1
#                        z[i] -= 2
#                        z[i+1] += 1
                    
                    h[i] -= 1           # update heights
 #                       z[i] = h[i] - h[i+1]
                    z_th[i] = rn.choice(thresholds_z) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation 
                    
                    ava_size += 1
                    if i == len(sites)-1:
                        if steady_first_time == False:
                            itsteady = added_grains+1
                            steady_first_time = True
                            return [heights, itsteady]
                        break
                    h[i+1] += 1   
#                    elif i != 0 and (i != len(sites)-1):
##                        z[i] -= 2
##                        z[i+1] += 1
##                        z[i-1] += 1
#                        
#                        
#                        h[i] -= 1           # update heights
#                        h[i+1] += 1                
#                        
#                        z_th[i] = rn.randint(1,2) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation
#                        
#                        counter5 += 1
#                        
#                    elif (i == len(sites)-1):
##                       z[i] -= 1
##                       z[i-1] += 1
#                       
#                       h[i] -= 1           # update heights  (different: grain leaves the system)              
#                       
#                       z_th[i] = rn.randint(1,2) #np.random.choice([1,2], p = [0.5, 0.5]) # change threshold upon relaxation
#                       
#                       counter5 += 1
                    
                i += 1
                #if h[0] != 0:    # don't break the loop at the beginning
#                if z[i] <= z_th[i]:   
#                    counter += 1
#            counter = 0            
#            for i in range(0, len(sites)): # exit condition linked with the fact that system is fully relaxed
#                if z_th[i] >= z[i]:
#                    counter += 1
            if counter == 0:
                break
#            if np.all((z_th - z) > np.zeros(len(sites))) == True:
#                break
#            print(added_grains)
        
        pile_heights[added_grains] = h[0]
        added_grains += 1
        
        if steady_first_time == True:
            ava_arr = np.append(ava_arr, ava_size)
        

        #print(float(counter2)/maxitime(L))
        
#        if counter2 == maxitime(L) and steady == 0:   # store iteration time taken to reach steady state
#            itsteady = counter2   # roughly steady
#            ## initialize height recording
#            site1h = np.zeros(itime-counter2+1)     # heights per time step
#            site1h[counter2-itsteady] = h[0]
#            steady = 1   # flip switch so this condition is no longer triggered
#
#        if counter2 > itsteady and itsteady != 0:
#            site1h[counter2-itsteady] = h[0]
#            #print(site1h)
        
        if added_grains == itime:       # implement exit condition if reached for all sites eg. for how many iterations I want it to run
            break
        

    
    return 


def av_crosst(system_size, mean_sample_size, itime, thresholds_z=[1,2], probs=[0.5,0.5]):
    
    zs = np.zeros(system_size)
    t_c = 0
    
    for i in range(mean_sample_size):
        x = oslo_crosst_revised(system_size, itime, thresholds_z, probs)
        zs += np.append(x[0][:-1] - x[0][1:], x[0][-1])
        t_c += x[1]
    av_z = np.sum(zs)/(float(mean_sample_size)*system_size)
    t_c /= float(mean_sample_size)
    
    return av_z, t_c
    

avs = [(1.495, 15.94),
 (1.545, 56.56),
 (1.58875, 217.32),
 (1.616875, 859.22),
 (1.65625, 3456.42),
 (1.685625, 13956.02),
 (1.6996875, 55983.18),
 (1.71265625, 225036.56)]    # average values



import matplotlib.pyplot as plt
plt.figure(4)
theoretical = np.zeros(len(avs))
actuals = np.zeros(len(avs))
for i in range(len(avs)):
    actuals[i] = avs[i][1]
    L = int(2**(i+2))
    theoretical[i] = avs[i][0]*0.5*(L*L + L)

#### compute some statistical tests to check if results are corroborated

from scipy.stats import pearsonr
pearson_r = round(pearsonr(theoretical, actuals)[0], 8)

sizes = [4,8,16,32,64,128,256,512]

y = np.linspace(1, max(actuals)+100000, 1000)
plt.loglog(y,y, '--', label= r'line $ y = x $')
plt.xlabel(r'theoretical $\langle t_c(L) \rangle = \frac{\langle z \rangle}{2}L^{2}\left(1+\frac{1}{L} \right) $', fontsize= 14)
plt.ylabel(r'numerical $\langle t_c(L) \rangle$', fontsize=14)
plt.loglog(theoretical, actuals, 'ro', label='Data points, pearsons r = {0}'.format(pearson_r))

ax = plt.figure(4).add_subplot(111)
counter = 0
x = [theoretical[i]+4**(i+1) for i in range(len(theoretical))]
y = [actuals[i]-2**(i+1) for i in range(len(theoretical))]

for xy in zip(x, y):                                       
    ax.annotate('L = %s' % sizes[counter], xy=xy, textcoords='data')
    counter += 1

plt.legend(loc=0, framealpha=0.8, prop={'size':10})
plt.show(4)
plt.close(4)
#with open("test.txt", "rb") as fp:   # Unpickling
#    b = pickle.load(fp)

delta = abs(theoretical-actuals)/theoretical
plt.figure(5)
plt.plot(sizes, delta, '--')
plt.plot(sizes, delta, 'ro', label = 'Data points, mean, std =  {0}, {1}'.format(round(np.mean(delta),5), round(np.std(delta), 5)))
plt.xlabel(r'system sizes $L$', fontsize=14)
plt.ylabel(r'$\Delta t_c = \frac{\left| \langle t_c(L) \rangle_{theo}-\langle t_c(L) \rangle_{num} \right|}{\langle t_c(L) \rangle_{theo}}$', fontsize=14)
plt.legend()
plt.show()
plt.close(5)


#big_list = []
#for i in sizes:
#    with open("system_size_take2_%s.txt" % i, "rb") as fp:   # Unpickling
#        big_list.append(pickle.load(fp))

haverages = []
for i in range(len(sizes)):
    haverages.append(np.average(big_list[i][0][int(big_list[i][2]):]))
sizes = np.array(sizes)
plt.figure(6)
plt.plot(sizes, haverages*1/sizes, 'ro', label='Data for different system sizes')

from scipy.optimize import curve_fit

heights = haverages*1/sizes
f = lambda L, a0, a1, omega: a0*(1-a1*L**(-1*omega))
fit = curve_fit(f, sizes, heights) 
plt.plot(np.linspace(1, 600, 1000), f(np.linspace(1, 600, 1000), fit[0][0], fit[0][1], fit[0][2]), 'b--', label=r'Fitted: $\frac{\langle h(t;L)\rangle_t}{L} = a_0 (1-L^{-\omega_1})$')
plt.xlabel(r'System sizes $L$', fontsize=14)
plt.ylabel(r'$\frac{\langle h(t;L)\rangle_t}{L}$', fontsize=14)
plt.legend(loc=0, framealpha=0.8, prop={'size':10})
plt.show(6)
plt.close(6)

plt.figure(7)
func = lambda L, a0, a1, omega: a0*L*(1-a1*L**(-1*omega))
plt.plot(sizes, heights*sizes, 'ro', label='for various L' )
plt.plot(np.linspace(1, 600, 1000), func(np.linspace(1, 600, 1000), fit[0][0], fit[0][1], fit[0][2]), 'b-', label=r'Fitted: $\langle h(t;L)\rangle_t} = a_0L(1-L^{-\omega_1})$')
plt.legend(loc=0, framealpha=0.8, prop={'size':10})
plt.xlabel(r'$L$', fontsize=14)
plt.ylabel(r'$\langle h(t;L)\rangle_t$', fontsize=14)
plt.close(7)

hsquares = []
for i in range(len(sizes)):
    squared = big_list[i][0][int(big_list[i][2]):]*big_list[i][0][int(big_list[i][2]):]
    hsquares.append(np.average(squared))            # steady state pile heights

sigma = np.sqrt(np.array(hsquares) - np.array(haverages)*np.array(haverages))
plt.figure(8)

plt.loglog(sizes, sigma, 'ro', label='for various L')
plt.ylabel(r'$\sigma_h$', fontsize=16)
plt.xlabel(r'$L$', fontsize=16)
o = lambda L,b,c: b*L**(-c)
sfit = curve_fit(o, sizes, sigma)
lspace = np.linspace(1, 600)
plt.loglog(lspace, o(lspace, sfit[0][0], sfit[0][1]), 'b-', label = r'$\sigma_h$ as a power law $bL^{\gamma}$')
plt.legend(loc=0, framealpha=0.8, prop={'size':10})
plt.show(8)
plt.close(8)

#with open("system_size_take2_1024.txt", "rb") as fp:   # Unpickling
#    pickle.load(fp)

plt.figure(9)
for i in range(len(sizes)):
    m = logbin(np.array(big_list[i][1], dtype='int64'), scale=1.25)
    plt.loglog(m[0], m[1], '-', label='L = {0}'.format(2**(i+2)))
    plt.legend()
plt.xlabel(r'avalanche size $s$')
plt.ylabel(r'$\tilde{P}_N\left(s;L\right)$', fontsize=14)
#plt.close(9)
tau = 1.554
D = 1/(2-tau)
plt.figure(10)
for i in range(1,len(sizes)):
    m = logbin(np.array(big_list[i][1], dtype='int64'), scale=1.25)
    plt.loglog(m[0]/((2**(i+2))**D), (m[0]**(tau))*m[1], '-', label='L = {0}'.format(2**(i+2)))
plt.legend(loc=0, framealpha=0.4)
plt.xlabel(r'$\frac{s}{L^D}$', fontsize=14)
plt.ylabel(r'$s^{-\tau_s}\tilde{P}_N\left(s;L\right)$', fontsize=14)
plt.close(10)


############# task 3b ############# computing tau_s
from scipy.stats import linregress
i = 7
plt.figure(14)
m = logbin(np.array(big_list[i][1], dtype='int64'), scale=1.25)
plt.loglog(m[0], m[1], 'o', markerfacecolor="None", markeredgewidth=1, label='L = {0}'.format(2**(i+2)))
plt.xlabel(r'avalanche size $s$')
plt.ylabel(r'$\tilde{P}_N\left(s;L\right)$', fontsize=14)
lspace = np.linspace(0,16,1000)
best_fit = linregress(m[0][7:-15], m[1][7:-15])
plt.loglog(np.exp(lspace), np.exp((lspace*best_fit[0]+best_fit[1]*np.ones(len(lspace)))), 'k-', 
           label=r'slope $\tau_s$ {0} $\pm$ {1}'.format(round(best_fit[0],4), round(best_fit[4], 4)) )
plt.legend()
#plt.close(14)

############### task 3c ################


plt.figure(11)
ks = range(1,5)             # choose k moments to calculate
sizes = [4,8,16,32,64,128,256,512]
moms_syss = []
for i in range(len(sizes)):
    moms = []
    for k in ks:
        kth_mom = np.sum(big_list[i][1] ** k)/len(big_list[i][1])
        moms.append(kth_mom)
    moms_syss.append(moms)
grads = []
moms_syss = np.array(moms_syss)
sizes = np.array(sizes)

### fit for the last three Ls as L >> 1 and s >> 1

for i in range(len(ks)):
    fit = linregress(np.log(sizes)[-3:], np.log(moms_syss[-3:,i]))
    grads.append((fit[0], fit[1]))  # slopes and intercepts
lspace = np.linspace(0,7,1000)
for i in range(len(ks)):
    plt.loglog(sizes, moms_syss[:,i], '.', label='k={0}'.format(i+1))
    plt.loglog(np.exp(lspace), np.exp((lspace*grads[i][0]+grads[i][1]*np.ones(len(lspace)))), 'k-')
plt.legend()
plt.xlabel('$L$', fontsize=16)
plt.ylabel(r'$\langle s^k \rangle$', fontsize=16)
#plt.close(11)

#### compute D and \tau_s using moment scaling analysis

plt.figure(12)
line = linregress(ks, np.array(grads)[:,0])
plt.plot(ks, np.array(grads)[:,0], 'k.', label = 'slope = {0} \n intercept = {1}'.format(round(line[0], 3), round(line[1], 3)))
kspace = np.linspace(0, 10, 1000)
plt.plot(kspace, (kspace*line[0]+line[1]*np.ones(len(kspace))))
plt.legend(loc=0)
plt.xlabel('$k$', fontsize=16)
plt.ylabel(r'$D\left(1+k- \tau_s \right)$', fontsize=16)

#plt.close(12)
