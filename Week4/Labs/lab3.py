import numpy as np
import matplotlib.pyplot as plt


#%% #1
# binom.pmf(k=["Yoda", "Vader", "R2D2"],n=3, p=[0.02,0.28,0.7])
plt.bar(["Yoda", "Vader", "R2D2"],[0.02,0.28,0.7], color="darkorange")
#%% #2 
pmf_mean = 15*0.02 + 4*0.28 + 1*0.7 # 2.12
print("PMF Mean:", pmf_mean)
#%% #3 
sim = np.random.choice([15,4,1], size=500, p=[0.02,0.28,0.7])
print("Simulated Mean:", sim.mean())

#%% #4 
# $5
np.std(sim) + sim.mean() #??
#%% #5 
variance = np.var(sim)
print("Simulated Var:", variance)
np.sqrt(variance)
#%% #6 
pmf_var = 0.02*(15-pmf_mean)**2 + 0.28*(4-pmf_mean)**2 + 0.70*(1-pmf_mean)**2
print("PMF MEAN:", pmf_var)

#%% #7 

#%% #8 

#%% #9 

#%% #10 

#%% #11 

