import numpy as np
from scipy import sparse
from scipy import optimize
import matplotlib.pyplot as plt

def TransitionMatrix(p,time,size):

    transition_matrix = sparse.lil_matrix((size,size),dtype='float')

    for i in range(-time,time + 1, 2):
         
        actual_pos = size//2 + i
        partial_transition = sparse.lil_matrix((size,size),dtype='float')

        if time != 0:
            alpha = (2*p - 1)/time
            prob_plus = (1/2)*(1 + alpha*i)
            prob_minus = (1/2)*(1 - alpha*i)
        else:
            prob_plus = p
            prob_minus = 1 - p

        if (1-prob_plus-prob_minus) > 10**(-10): print('Error!\n')

        if i < size//2 and i > -size//2:
            partial_transition[actual_pos + 1, actual_pos] = prob_plus
            partial_transition[actual_pos - 1, actual_pos] = prob_minus
        elif i == size//2:
            partial_transition[-size//2 , actual_pos] = prob_plus
            partial_transition[actual_pos - 1, actual_pos] = prob_minus
        elif i == -size//2:
            partial_transition[actual_pos + 1, actual_pos] = prob_plus
            partial_transition[size//2, actual_pos] = prob_minus

        transition_matrix = transition_matrix + partial_transition

    return transition_matrix

def general_variance(x,a,b,c,d):
        return a*x**3+b*x**2+c*x+d

def Simulation(q,p,size):

    variance_list = []
    time_vector = []
    probability_vector = sparse.lil_matrix((lattice_size,1),dtype='float')
    probability_vector[lattice_size//2,0] = 1
    transition_matrix = TransitionMatrix(q,0,size)

    for t in range(0,size//2):

        print(' time = ',t,end='\r')

        variance = 0
        for l in range(-size//2,size//2):
            variance = variance + l**(2)*probability_vector[size//2 + l,0]

        variance_list.append(variance)

        time_vector.append(t)
        probability_vector = np.dot(transition_matrix,probability_vector)
        transition_matrix = TransitionMatrix(p,t+1,size)

    return (variance_list, probability_vector, time_vector)

q = 0.5
p = 1
lattice_size = 801

variance, probability_vector, time_vector = Simulation(q,p,lattice_size)
probability_vector = probability_vector.todense()

position_vector = []
for i in range(-lattice_size//2,lattice_size//2):
    position_vector.append(i)

position_vector = np.array(position_vector)
time_vector = np.array(time_vector)
variance = np.array(variance)

fit_params, pcov = optimize.curve_fit(general_variance,time_vector,variance)
a = round(fit_params[0],5)
b = round(fit_params[1],5)
c = round(fit_params[2],5)
d = round(fit_params[3],5)

fig = plt.figure(figsize=(16,9),dpi=200) 
title_str = 'q = '+str(q)+', p = '+str(p)
plt.title(title_str,fontsize=16)
l, = plt.plot(time_vector,variance,label = 'Simulation',lw=2)
fit_label = str(a)+r'$t^{3}$'+ '+' +str(b)+r'$t^{2}$'
fit_label = fit_label + '+' + str(c)+r'$t$' + '+' + str(d)
m, = plt.plot(time_vector,general_variance(time_vector,*fit_params),label = fit_label,ls='--')
plt.grid(linestyle='--')
plt.xlabel(r't',fontsize=16)
plt.ylabel(r'$\sigma^{2}$(t)',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(handles=[l,m],fontsize=14)
plt.savefig('variance',bbox_inches='tight')
plt.clf()

fig = plt.figure(figsize=(16,9),dpi=200) 
plt.title(title_str+r'$, t ='+str(lattice_size//2-1)+'$',fontsize=16)
k,= plt.plot(position_vector,probability_vector,lw=2,label='Simulation')
plt.grid(linestyle='--')
plt.xlabel(r'$x$',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(r'$Pr(x)$',fontsize=16)
plt.legend(handles=[k],fontsize=14)
plt.savefig('position_distribuition',bbox_inches='tight')
