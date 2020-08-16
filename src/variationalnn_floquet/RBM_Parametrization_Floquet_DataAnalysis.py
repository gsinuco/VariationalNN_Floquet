#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:21:04 2020

@author: german
"""
import numpy as np
from matplotlib import pyplot as plt
import model as Model
from scipy import stats


A = Read_RBM("stats_11-20.dat",Stats="stats")
#%%
def Read_Psi_UF(file_,dim,b):
    UF_real = np.loadtxt(file_,max_rows=dim)
    UF_imag = np.loadtxt(file_,max_rows=dim)
    loss    = np.loadtxt(file_,max_rows=1)
            
    UF = np.zeros([dim,b],dtype=complex)
    UF = UF_real+np.complex(0.0,1.0)*UF_imag        
    return UF,loss


def RBM_DataAnalysis(filename='RBM_TrainingFloquet_WaveFunction_1stRun.dat'):
#script_params ='#RWA Training: keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999,                                      epsilon=1e-07, amsgrad=False,name=Adam)'+ str([N_training_RWA, N_tr_ph, N_tr_rho])+ 'Matrix Diagonalisation: optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,                                          epsilon=1e-07, amsgrad=False,name=Adam)'+str(N_training_Matrix)

#####################################
## 1-10 runs 
# hideen_ = 8
# hidden  = 16
#        if(i_<=1024 and np.mod(i_,64)==0):
#        if(i_>1024 and np.mod(i_,256)==0):
#        if(i_ == (N_training-1)):

## 11-20 runs
# hideen_ = 2
# hidden  = 4
#        if(i_<=1024 and np.mod(i_,64)==0):
#        if(i_>1024 and np.mod(i_,256)==0):
#        if(i_ == (N_training-1)):

## 21  run 
# hideen_ = 8
# hidden  = 16
# the previously traineed state is ussed for the next 
# field configuration.
# the model is trained against the RWA only for the first 
# iteration 
#        if(i_<=1024 and np.mod(i_,64)==0):
#        if(i_>1024 and np.mod(i_,256)==0):
#        if(i_ == (N_training-1)):

 
#filename = 'RBM_TrainingFloquet_WaveFunction_1stRun.dat'
#filename = 'RBM_TrainingFloquet_WaveFunction_6stRun.dat'
    file_ = open(filename,'r')

    script_params = file_.readline()
    RWA_optimiser_config        = script_params[:158]
    RWA_trainingLoop_config     = script_params[158:167]
    Floquet_optimiser_config    = script_params[167:355]
    Floquet_trainingLoop_config = script_params[355:361]


    N_training_RWA = 4
    N_training_Matrix = 18394
    N = 32

    iteration = []
    loss_list = []
    fidelity_list = []
    RBM_Floquet_spectrum_up   = []
    RBM_Floquet_spectrum_down = []

    Floquet_spectrum_up   = []
    Floquet_spectrum_down = []
    

    for i in range(N):

        delta = 1.7
        Omega = 10.0*i/N
        phase = np.arctan(1.0)/4.1341341
        
        model = Model.RBM_Model(delta,Omega,phase)
        
        Floquet_spectrum_down = np.concatenate([Floquet_spectrum_down,  [model.E_Floquet[0]]],axis=0)
        Floquet_spectrum_up   = np.concatenate([Floquet_spectrum_up,  [model.E_Floquet[1]]],axis=0)

        index_ = 0
        
        N_training        = N_training_RWA
        N_training_Matrix = 18394 + int(np.power((1.0*i/N),6)*4*512.0)    

        if (i == 0 ): 
            model_params  = np.loadtxt(file_,max_rows=11)
            dim     = int(model_params[5]*(2*model_params[6]+1))
            UF_dim  = int(model_params[5]*(2*model_params[10]+1))
            S_local = int(model_params[5])
        if (i != 0 ): 
            model_params  = np.loadtxt(file_,max_rows=12)
            dim     = int(model_params[6]*(2*model_params[7]+1))
            UF_dim  = int(model_params[6]*(2*model_params[11]+1))
            S_local = int(model_params[6])

    # READ THE FLOQUET STATES GUESS AND INITIAL LOSS
        UF_real = np.loadtxt(file_,max_rows=dim)
        UF_imag = np.loadtxt(file_,max_rows=dim)
        loss    = np.loadtxt(file_,max_rows=1)

        UF = np.zeros([dim,S_local],dtype=complex)
        UF = UF_real+np.complex(0.0,1.0)*UF_imag
        U_Floquet  = model.U_Floquet.numpy()
    
        projection    = 1-np.trace(np.abs(np.transpose(np.conjugate(U_Floquet))@UF))/2.0
        fidelity_list = np.concatenate([fidelity_list,[projection]],axis=0)
    
        N_training = N_training_Matrix

        i__ = 0
        for i_ in range(N_training):
        
            if(i_<=1024 and np.mod(i_,64)==0):

                UF_real = np.loadtxt(file_,max_rows=dim)
                UF_imag = np.loadtxt(file_,max_rows=dim)
                loss    = np.loadtxt(file_,max_rows=1)
            
                UF = np.zeros([dim,S_local],dtype=complex)
                UF = UF_real+np.complex(0.0,1.0)*UF_imag        

                if(i_==0): loss_list     = np.concatenate([loss_list,[loss]],axis=0)
                U_            = np.transpose(np.conjugate(UF))@model.H_TLS@UF        
                projection    = 1-np.trace(np.abs(np.transpose(np.conjugate(U_Floquet))@UF))/2.0
                #fidelity_list = np.concatenate([fidelity_list,[projection]],axis=0)
                if(i==0): iteration = np.concatenate([iteration,[i_]],axis=0)
            

            if(i_>1024 and np.mod(i_,256)==0):
                UF,loss   = Read_Psi_UF(file_,dim,S_local) 
                #loss_list = np.concatenate([loss_list,[loss]],axis=0)
                U_        = np.transpose(np.conjugate(UF))@model.H_TLS@UF        
                projection = 1 - np.trace(np.abs(np.transpose(np.conjugate(U_Floquet))@UF))/2.0
                #fidelity_list = np.concatenate([fidelity_list,[projection]],axis=0)
                if(i==0): iteration = np.concatenate([iteration,[i_]],axis=0)

            if(i_ == (N_training-1)):
                UF,loss   = Read_Psi_UF(file_,dim,S_local) 
                loss_list = np.concatenate([loss_list,[loss]],axis=0)
                U_        = np.transpose(np.conjugate(UF))@model.H_TLS@UF        
                projection = 1-np.trace(np.abs(np.transpose(np.conjugate(U_Floquet))@UF))/2.0
                fidelity_list = np.concatenate([fidelity_list,[projection]],axis=0)
                if(i==0): iteration = np.concatenate([iteration,[i_]],axis=0)
                RBM_Floquet_spectrum_down    = np.concatenate([RBM_Floquet_spectrum_down,[np.real(U_[index_   , index_  ])]], axis = 0)
                RBM_Floquet_spectrum_up  = np.concatenate([RBM_Floquet_spectrum_up,[np.real(U_[index_+1 , index_+1])]], axis = 0) 
            

        
        #file_.close()
    return iteration, fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down


#%%
N = 32
Omega = (10.0/N)*(np.linspace(1,N,N)-1)
#runs = np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.int)
#runs = np.array([11,12,13,14,15,16,17,18,19,20],dtype=np.int)
runs = np.array([21,22,23],dtype=np.int)

#datos_ = RBM_DataAnalysis("RBM_TrainingFloquet_WaveFunction_1stRun.dat")    
#datos = np.zeros_like(datos_)

Energy_error  = np.zeros([runs.size,Omega.size],dtype=np.float64)
Fidelity       = np.zeros([runs.size,Omega.size],dtype=np.float64)
Loss_error    = np.zeros([runs.size,Omega.size],dtype=np.float64)
Loss_Initial  = np.zeros([runs.size,Omega.size],dtype=np.float64)

for i in range(3):#runs: 
    #filename = "RBM_TrainingFloquet_WaveFunction_"+ str(runs[i]) + "stRun.dat"
    #filename = "RBM_TrainingFloquet_WaveFunction_"+ str(runs[i]) + "thRun.dat"
    filename = "RBM_TrainingFloquet_WaveFunction_"+ str(runs[i]) + "rdRun.dat"

    #iteration,fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down = RBM_DataAnalysis(filename)
    datos_ = RBM_DataAnalysis(filename)    
    Energy_error[i]  = np.abs(np.abs(datos_[3]-datos_[4]) - np.abs(datos_[5]-datos_[6]))
    Fidelity[i]      = 1-np.abs(datos_[1][1::2])
    Loss_error[i]    = np.abs(datos_[2][1::2])
    Loss_Initial[i]  = np.abs(datos_[2][0::2])
    

#%%

statistics_errorE_C   = stats.describe(Energy_error)
statistics_errorF_C   = stats.describe(Loss_error)
statistics_errorI_C   = stats.describe(Loss_Initial)
statistics_fidelity_C = stats.describe(Fidelity)
#%%
RBM_stats_ = [statistics_errorE_B,statistics_errorF_B,statistics_errorI_B,statistics_fidelity_B]
#save = statistics_save("stats_11-20.dat",RBM_stats=RBM_stats_)  


#A = Read_RBM("stats_11-20.dat",Stats="stats")
#%%



f_ = plt.figure
#%%
plt.subplot(311)
plt.errorbar(Omega,statistics_errorE[2], np.sqrt(statistics_errorE[3]),barsabove=True, ls="-",marker=".",label="test")
plt.plot(Omega,statistics_errorE[2], ".-")
#plt.plot(Omega,statistics_errorE[1][0])
#plt.fill_between(Omega, statistics_errorE[1][0], statistics_errorE[1][1], color='#539ecd')

#plt.yscale("log")
#plt.xlabel("Omega")
plt.xlim(0,10)
plt.xticks(range(11), ())
plt.ylabel('$\Delta E$')
#plt.ylabel("Avg. Energy error")
#plt.show()

plt.subplot(312)
#plt.plot(Omega,statistics_fidelity[2])
#plt.plot(Omega,statistics_overlap[1][0])
#plt.plot(Omega,statistics_overlap[1][1])
#plt.plot(Overlap_error
plt.errorbar(Omega,statistics_fidelity[2], np.sqrt(statistics_fidelity[3]),barsabove=True, ls="-",marker=".",label="test")
#plt.fill_between(Omega, statistics_fidelity[1][0], statistics_fidelity[1][1], color='#539ecd')
plt.plot(Omega,statistics_fidelity[2], ".-")
#plt.errorbar(Omega,statistics_fidelity[2], yerror=np.sqrt(statistics_fidelity[3]))
#plt.yscale("log")
#plt.xlabel("Omega")
plt.xlim(0,10)
plt.xticks(range(11), ())
#plt.ylabel("Avg. Overlap error")
plt.ylabel('Fidelity')
#plt.show()

plt.subplot(313)
#plt.fill_between(Omega, statistics_errorF[1][0], statistics_errorF[1][1], color='#539ecd')
#plt.fill_between(Omega, statistics_errorI[1][0], statistics_errorI[1][1], color='#509ecd')
plt.plot(Omega,statistics_errorI[2], ".-")
plt.plot(Omega,statistics_errorF[2], ".-")
plt.errorbar(Omega,statistics_errorI[2], np.sqrt(statistics_errorI[3]),barsabove=True, ls="-",marker=".",label="test")
plt.errorbar(Omega,statistics_errorF[2], np.sqrt(statistics_errorF[3]),barsabove=True, ls="-",marker=".",label="test")

#plt.plot(Omega,Loss_Initial, ".-")
plt.xlabel('$\Omega$')
#plt.xlabel("Omega")
plt.xlim(0,10)
plt.xticks(range(11))
#plt.ylabel("Avg. Loss error")
plt.ylabel('$\Gamma$')
#plt.show()

#plt.tight_layout()
plt.show()
#%%
iteration,fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down = RBM_DataAnalysis('RBM_TrainingFloquet_WaveFunction_10stRun.dat')

#%%

plt.plot(loss_list,".-")
plt.plot(1600*fidelity_list, ".-")
#plt.ylim(-10,200)
plt.xlim(1500,1700)
plt.show()
#%%
plt.plot(Omega,np.abs(RBM_Floquet_spectrum_up-RBM_Floquet_spectrum_down), ".-")
plt.plot(Omega,Floquet_spectrum_up-Floquet_spectrum_down, ".-")
plt.yscale('log')
plt.xlabel("Driving amplitude")
plt.ylabel("Qubit gap")
plt.show()


plt.plot(iteration,loss_list[1703:1788],".-")
plt.plot(iteration,loss_list[1789:1874],".-")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()



#%%

def statistics_save(file,**kwargs):    
    # script params: a mixed arrasy
     if(RBM_stats != None):
         file_ = open(file,'w')
         for i in range(4):
             #np.savetxt(file_,RBM_stats[i][0])
             np.savetxt(file_,RBM_stats[i][1][0])
             np.savetxt(file_,RBM_stats[i][1][1])
             np.savetxt(file_,RBM_stats[i][2])
             np.savetxt(file_,RBM_stats[i][3])
             np.savetxt(file_,RBM_stats[i][4])
             np.savetxt(file_,RBM_stats[i][5])
             
         file_.close()
     return "done"


def Read_RBM(file,**kwargs):
    # model: class Model
    # script params: a mixed arrasy
    #                  script_params = ["""RWA Training: keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999,epsilon=1e-07, amsgrad=False,name='Adam')""",N_training, N_tr_ph, N_tr_rho,tf, """Matrix Diagonalisation: optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,                                          epsilon=1e-07, amsgrad=False,name='Adam')""",N_training_matrix]  
    # script_result = [model.trainable_variables,loss_value] 
    #
    #
    model          = kwargs.get('Model',None)
    script_params  = kwargs.get('Script_Params',None)
    script_results = kwargs.get('Script_Results',None)
    stats          = kwargs.get('Stats',None)
    
    if(stats!=None):
        file_ = open(file,'r')
        stats_ = np.loadtxt(file_)
        errorE   = stats_[0:6]
        errorF   = stats_[6:12]
        errorI   = stats_[12:18]
        fidelity = stats_[18:24]

        return "r",errorE,errorF,errorI,fideltiy
        

    if(model != None):
        file_ = open(file,'r')
        model_params = [model.spin,model.omega_0,model.delta,model.omega,model.Omega,
                       model.phase,model.S,model.N,model.dim,model.hidden_n,
                       model.hidden_ph,model.N_Floquet_UF]
        model_params = np.loadtxt(file_,model_params)
        
        file_.close()
        return "done"
    
    
    if(script_params != None):
        file_ = open(file,'r')
        file_.write(script_params)
        file_.close()
        return "done"

    if(script_results != None):        
        file_ = open(file,'r')
        UF_real = np.zeros([...],dtype=float64)
        #UF_real = np.loadtxt(file_,maxrows=)
        UF_imag = np.loadtxt(file_)
        loss_   = np.loadtxt(file_)

        #for i in range(script_results[0][0].shape[0]):
        #    np.savetxt(file_,script_results[0][0][i,:])                    
        #np.savetxt(file_,script_results[0][1][:])        
        #np.savetxt(file_,script_results[0][2][:])                  
        
        #for i in range(script_results[0][3].shape[0]):
        #    np.savetxt(file_,script_results[0][3][i,:])                    
        #np.savetxt(file_,script_results[0][4][:])        
        #np.savetxt(file_,script_results[0][4][:])        
        
        #np.savetxt(file_,[script_results[1]])        
        
        file_.close()
        return [UF_real,UF_imag,loss_]
        
     
def Read_Psi_UF(file_,dim,b):
    UF_real = np.loadtxt(file_,max_rows=dim)
    UF_imag = np.loadtxt(file_,max_rows=dim)
    loss    = np.loadtxt(file_,max_rows=1)
            
    UF = np.zeros([dim,b],dtype=complex)
    UF = UF_real+np.complex(0.0,1.0)*UF_imag        
    return UF,loss
    
