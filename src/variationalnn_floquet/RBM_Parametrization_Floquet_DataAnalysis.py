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




def statistics_save(file,RBM_stats):#**kwargs):    
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


def statistics_save_GNUPLOT(file,RBM_stats):#**kwargs):    
    # script params: a mixed arrasy
     N=32
     if(RBM_stats != None):
         data_ = np.zeros([32,3],dtype=np.double)
         data_[:,0] = (10.0/N)*(np.linspace(1,N,N)-1)
         file_ = open(file,'w')
         for i in range(4):
             # mean and variance
             #np.savetxt(file_,RBM_stats[i][2])
             #np.savetxt(file_,RBM_stats[i][3])
             data_[:,1] = RBM_stats[i][2]
             data_[:,2] = RBM_stats[i][3]
             np.savetxt(file_,data_)
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

        return "r",errorE,errorF,errorI,fidelity
        

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
    

#A = Read_RBM("stats_11-20.dat",Stats="stats")
#%%


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
    
    #file_psi = open(filename+'comparison_3D.dat','w')

    for i in range(N):

        iteration = []
        loss_list = []
        fidelity_list = []

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
        loss_list = np.concatenate([loss_list,[loss]],axis=0)
        iteration = np.concatenate([iteration,[0]],axis=0)
    
        N_training = N_training_Matrix

        i__ = 0
        for i_ in range(N_training):
        
            if(i_<=1024 and np.mod(i_,64)==0):

                UF_real = np.loadtxt(file_,max_rows=dim)
                UF_imag = np.loadtxt(file_,max_rows=dim)
                loss    = np.loadtxt(file_,max_rows=1)
            
                UF = np.zeros([dim,S_local],dtype=complex)
                UF = UF_real+np.complex(0.0,1.0)*UF_imag        

                #if(i_==0): loss_list     = np.concatenate([loss_list,[loss]],axis=0)
                U_            = np.transpose(np.conjugate(UF))@model.H_TLS@UF        
                #U_            = np.mod(np.real(U_),model.omega) #- model.omega/2.0
                projection    = 1-np.trace(np.abs(np.transpose(np.conjugate(U_Floquet))@UF))/2.0
                #fidelity_list = np.concatenate([fidelity_list,[projection]],axis=0)
                #loss_list     = np.concatenate([loss_list,[loss]],axis=0)
                iteration = np.concatenate([iteration,[i_]],axis=0)
                #if(i==0): iteration = np.concatenate([iteration,[i_]],axis=0)
            
            

            if(i_>1024 and np.mod(i_,256)==0):
                UF,loss   = Read_Psi_UF(file_,dim,S_local) 
                U_        = np.transpose(np.conjugate(UF))@model.H_TLS@UF        
                #U_            = np.mod(np.real(U_),model.omega) #- model.omega/2.0
                projection = 1 - np.trace(np.abs(np.transpose(np.conjugate(U_Floquet))@UF))/2.0
                #fidelity_list = np.concatenate([fidelity_list,[projection]],axis=0)
                #loss_list = np.concatenate([loss_list,[loss]],axis=0)
                iteration = np.concatenate([iteration,[i_]],axis=0)
                #if(i==0): iteration = np.concatenate([iteration,[i_]],axis=0)

            if(i_ == (N_training-1)):
                UF,loss   = Read_Psi_UF(file_,dim,S_local) 
                U_        = np.transpose(np.conjugate(UF))@model.H_TLS@UF        
                #U_            = np.mod(np.real(U_),model.omega) #- model.omega/2.0
                projection = 1-np.trace(np.abs(np.transpose(np.conjugate(U_Floquet))@UF))/2.0
                fidelity_list = np.concatenate([fidelity_list,[projection]],axis=0)
                loss_list = np.concatenate([loss_list,[loss]],axis=0)
                iteration = np.concatenate([iteration,[i_]],axis=0)
                #if(i==0): iteration = np.concatenate([iteration,[i_]],axis=0)
                RBM_Floquet_spectrum_down    = np.concatenate([RBM_Floquet_spectrum_down,[np.real(U_[index_   , index_  ])]], axis = 0)
                RBM_Floquet_spectrum_up  = np.concatenate([RBM_Floquet_spectrum_up,[np.real(U_[index_+1 , index_+1])]], axis = 0) 
        #if i == 16:        
        #print('RBM:',i,1-projection, np.argmax(np.abs(UF[:,0]))//2-16,np.argmax(np.abs(UF[:,1]))//2-16)
        #print('Flo:',i,1-projection, np.argmax(np.abs(model.U_Floquet[:,0]))//2-16,np.argmax(np.abs(model.U_Floquet[:,1]))//2-16)
        #print(np.diag(np.real(U_)))
        #print(np.mod((np.real(U_)),model.omega))
        #print(((np.real(U_)) - np.mod(np.diag(np.real(U_)),model.omega))/model.omega)
        
        #plt.figure()
        #plt.plot(np.abs(UF[:,0]))
        #plt.plot(np.abs(UF[:,1]))
        #plt.title("RBM")
        #plt.xlabel("Channel Index")
        #plt.ylabel("|<i|F>|")
        #plt.show()

        #plt.figure()
        #plt.plot(np.abs(UF[:,0]))
        #plt.plot(np.abs(np.roll(UF[:,0],-2)))
        #plt.plot(np.abs(np.roll(UF[:,0],-1)))
        #plt.plot(np.abs(U_Floquet[:,0]))
        #plt.plot(np.abs(U_Floquet[:,1]))
        #plt.title("Floquet")
        #plt.xlabel("Channel Index")
        #plt.ylabel("|<i|F>|")
        #plt.show()
        
        #plt.figure()
        #plt.imshow(np.abs(U_))

        absPsi_comparison         = np.zeros([UF.shape[0],11],dtype=np.double)
        absPsi_comparison[:,0]    = np.linspace(0,UF.shape[0]-1,UF.shape[0],dtype=np.int)
        absPsi_comparison[:,1]    = np.mod(np.linspace(0,UF.shape[0]-1,UF.shape[0],dtype=np.int),2) - 0.5
        absPsi_comparison[:,2]    = absPsi_comparison[:,0]//2 - 16
        absPsi_comparison[:,3:5]  = Model.Rect2Pol(UF[:,0])
        absPsi_comparison[:,5:7]  = Model.Rect2Pol(UF[:,1])
        aux = np.zeros([66,1],dtype=np.complex)
        aux[:,0] = model.U_Floquet[:,0]
        absPsi_comparison[:,7:9]  = Model.Rect2Pol(aux)
        aux[:,0] = model.U_Floquet[:,1]
        absPsi_comparison[:,9:11] = Model.Rect2Pol(aux)        
        #file_psi.write('index |bare> photon_number rho_0_RBM phi_0_RBM rho_1_RBM phi_1_RBM rho_0_F phi_0_F rho_1_F phi_1_F')
        #np.savetxt(file_psi,absPsi_comparison)                
        #file_psi.write("\n\n")



    
        data_ = np.zeros([iteration.shape[0],3],dtype=np.double)
        #data_[:,0] = iteration
        #data_[:,1] = fidelity_list
        #data_[:,2] = loss_list
        #np.savetxt(file_psi,data_)
        #file_psi.write("\n\n")

        
    #file_psi.close()
    
    return iteration, fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down


#%%
N = 32
Omega = (10.0/N)*(np.linspace(1,N,N)-1)
iteration,fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down = RBM_DataAnalysis('RBM_TrainingFloquet_WaveFunction_9stRun.dat')
Omega = np.linspace(1,N,N)-1#(10.0/N)*(np.linspace(1,N,N)-1)

Dressed_manifold_energies=np.zeros([18,N],dtype=np.double)
counter = 0
for s in [-1,1]:
    for r in range(-4,5):    
        
        if s<0: 
            Dressed_manifold_energies[counter,:] = RBM_Floquet_spectrum_down + r*2.7
        if s>0:
            Dressed_manifold_energies[counter,:] = RBM_Floquet_spectrum_up   + r*2.7
        
        counter = counter + 1
        
Dressed_gaps = np.zeros([N],dtype=np.double)
for s in range(N):
    Dressed_gaps[s] = np.abs(Dressed_manifold_energies[np.argsort(np.abs(Dressed_manifold_energies[:,s]))[0],s] -
                             Dressed_manifold_energies[np.argsort(np.abs(Dressed_manifold_energies[:,s]))[1],s]) 
    
#%%plt.figure()


plt.figure()
plt.plot(Omega,Dressed_gaps/2)    
plt.plot(Omega,Floquet_spectrum_up + 2.0*2.7,'-',linewidth=3)
plt.plot(Omega,Floquet_spectrum_up + 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up + 0.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up - 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up - 2.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 2.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 0.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down - 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down - 2.0*2.7,'-')
plt.plot(Omega,RBM_Floquet_spectrum_up + 3.0*2.7, ".",label='up,+3')
plt.plot(Omega,RBM_Floquet_spectrum_up + 2.0*2.7, ".",label='up,+2')
plt.plot(Omega,RBM_Floquet_spectrum_up + 1.0*2.7, ".",label='up,+1')
plt.plot(Omega,RBM_Floquet_spectrum_up - 0.0*2.7, ".",label='up,+0')
plt.plot(Omega,RBM_Floquet_spectrum_up - 1.0*2.7, ".",label='up,-1')
plt.plot(Omega,RBM_Floquet_spectrum_up - 2.0*2.7, ".",label='up,-2')
plt.plot(Omega,RBM_Floquet_spectrum_up - 3.0*2.7, ".",label='up,-3')
plt.plot(Omega,RBM_Floquet_spectrum_down + 3.0*2.7, ".",label='down,+3')
plt.plot(Omega,RBM_Floquet_spectrum_down + 2.0*2.7, ".",label='down,+2')
plt.plot(Omega,RBM_Floquet_spectrum_down + 1.0*2.7, ".",label='down,+1')
plt.plot(Omega,RBM_Floquet_spectrum_down - 0.0*2.7, ".",label='down,+0')
plt.plot(Omega,RBM_Floquet_spectrum_down - 1.0*2.7, ".",label='down,-1')
plt.plot(Omega,RBM_Floquet_spectrum_down - 2.0*2.7, ".",label='down,-2')
plt.plot(Omega,RBM_Floquet_spectrum_down - 3.0*2.7, ".",label='down,-3')
plt.ylim(-2.7/2,2.7/2)
plt.xlabel("Driving amplitude")
plt.ylabel("Qubit gap")
plt.legend(loc='left')
plt.show()

#%%
N = 32
Omega = (10.0/N)*(np.linspace(1,N,N)-1)
iteration,fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down = RBM_DataAnalysis('RBM_TrainingFloquet_WaveFunction_12thRun.dat')
Omega = np.linspace(1,N,N)-1#(10.0/N)*(np.linspace(1,N,N)-1)

plt.figure()
plt.plot(Omega,Floquet_spectrum_up + 2.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up + 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up + 0.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up - 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up - 2.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 2.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 0.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down - 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down - 2.0*2.7,'-')
plt.plot(Omega,RBM_Floquet_spectrum_down + 2.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_down + 1.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_down - 0.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_down - 1.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_down - 2.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up + 2.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up + 1.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up - 0.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up - 1.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up - 2.0*2.7, ".")
plt.ylim(-2.7/2,2.7/2)
plt.xlabel("Driving amplitude")
plt.ylabel("Qubit gap")
plt.legend()
plt.show()

#%%
N = 32
Omega = (10.0/N)*(np.linspace(1,N,N)-1)
iteration,fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down = RBM_DataAnalysis('RBM_TrainingFloquet_WaveFunction_21rdRun.dat')
Omega = np.linspace(1,N,N)-1#(10.0/N)*(np.linspace(1,N,N)-1)

Dressed_manifold_energies=np.zeros([10,N],dtype=np.double)
counter = 0 

for s in [-1,1]:
    for r in range(-2,3):    
        if s<0: 
            Dressed_manifold_energies[counter,:] = Floquet_spectrum_down + r*2.7
        if s>0:
            Dressed_manifold_energies[counter,:] = Floquet_spectrum_up   + r*2.7
        counter = counter + 1
    
    
plt.figure()
plt.plot(Omega,Floquet_spectrum_up + 2.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up + 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up + 0.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up - 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_up - 2.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 2.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down + 0.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down - 1.0*2.7,'-')
plt.plot(Omega,Floquet_spectrum_down - 2.0*2.7,'-')
plt.plot(Omega,RBM_Floquet_spectrum_down + 2.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_down + 1.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_down - 0.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_down - 1.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_down - 2.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up + 2.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up + 1.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up - 0.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up - 1.0*2.7, ".")
plt.plot(Omega,RBM_Floquet_spectrum_up - 2.0*2.7, ".")
plt.ylim(-2.7/2,2.7/2)
plt.xlabel("Driving amplitude")
plt.ylabel("Qubit gap")
plt.show()

#%%

runs = np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.int)
#runs = np.array([11,12,13,14,15,16,17,18,19,20],dtype=np.int)
#runs = np.array([21,22,23],dtype=np.int)

#datos_ = RBM_DataAnalysis("RBM_TrainingFloquet_WaveFunction_1stRun.dat")    
#datos = np.zeros_like(datos_)

Energy_error  = np.zeros([runs.size,Omega.size],dtype=np.float64)
Fidelity       = np.zeros([runs.size,Omega.size],dtype=np.float64)
Loss_error    = np.zeros([runs.size,Omega.size],dtype=np.float64)
Loss_Initial  = np.zeros([runs.size,Omega.size],dtype=np.float64)

Dressed_manifold_energies = np.zeros([18,Omega.size],dtype=np.double)
Dressed_gaps              = np.zeros([runs.size,Omega.size],dtype=np.double)

for i in range(runs.shape[0]):#runs: 
    filename = "RBM_TrainingFloquet_WaveFunction_"+ str(runs[i]) + "stRun.dat"
    #filename = "RBM_TrainingFloquet_WaveFunction_"+ str(runs[i]) + "thRun.dat"
    #filename = "RBM_TrainingFloquet_WaveFunction_"+ str(runs[i]) + "rdRun.dat"

    #iteration,fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down = RBM_DataAnalysis(filename)
    datos_ = RBM_DataAnalysis(filename)    
    counter = 0
    RBM_Floquet_spectrum_down = datos_[5]
    RBM_Floquet_spectrum_up   = datos_[4]
    for s in [-1,1]:
        for r in range(-4,5):            
            if s<0: 
                Dressed_manifold_energies[counter,:] = RBM_Floquet_spectrum_down + r*2.7
            if s>0:
                Dressed_manifold_energies[counter,:] = RBM_Floquet_spectrum_up   + r*2.7
            counter = counter + 1
        
    
    Energy_error[i] = np.abs(
                      Dressed_manifold_energies[np.argsort(np.abs(Dressed_manifold_energies),axis=0)[0,:]] -
                      Dressed_manifold_energies[np.argsort(np.abs(Dressed_manifold_energies),axis=0)[1,:]]) 
        
    #Energy_error[i]  = np.abs(np.abs(datos_[3]-datos_[4]) - np.abs(datos_[5]-datos_[6]))
    Fidelity[i]      = 1-np.abs(datos_[1][1::2])
    Loss_error[i]    = np.abs(datos_[2][1::2])
    Loss_Initial[i]  = np.abs(datos_[2][0::2])


    

## 27 December 2020: let's collect all results, assuming (correctly, I think) that all
## runs where equivalents (i.e. used the same network parameters)
#%%
runs0 = np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.int)
runs1 = np.array([11,12,13,14,15,16,17,18,19,20],dtype=np.int)
runs2 = np.array([21,22,23],dtype=np.int)
Energy_error  = np.zeros([23,Omega.size],dtype=np.float64)
Fidelity      = np.zeros([23,Omega.size],dtype=np.float64)
Loss_error    = np.zeros([23,Omega.size],dtype=np.float64)
Loss_Initial  = np.zeros([23,Omega.size],dtype=np.float64)

run = False
if run:
    for i in range(1,11):
        print(i)
        if i > 0  and i <=10: 
            filename = "RBM_TrainingFloquet_WaveFunction_"+ str(i) + "stRun.dat"
        if i > 10 and i <=20: 
            filename = "RBM_TrainingFloquet_WaveFunction_"+ str(i) + "thRun.dat"
        if i > 20 and i <=23: 
            filename = "RBM_TrainingFloquet_WaveFunction_"+ str(i) + "rdRun.dat"

    #iteration,fidelity_list,loss_list,RBM_Floquet_spectrum_up,RBM_Floquet_spectrum_down,Floquet_spectrum_up,Floquet_spectrum_down = RBM_DataAnalysis(filename)
        datos_ = RBM_DataAnalysis(filename)    
        Energy_error[i-1]  = np.abs(np.abs(datos_[3]-datos_[4]) - np.abs(datos_[5]-datos_[6]))
        Fidelity[i-1]      = 1-np.abs(datos_[1][1::2])
        Loss_error[i-1]    = np.abs(datos_[2][1::2])
        Loss_Initial[i-1]  = np.abs(datos_[2][0::2])

#%%
statistics_errorE_C   = stats.describe(Energy_error)
statistics_errorF_C   = stats.describe(Loss_error)
statistics_errorI_C   = stats.describe(Loss_Initial)
statistics_fidelity_C = stats.describe(Fidelity)

#%%
#RBM_stats_ = [statistics_errorE_C,statistics_errorF_C,statistics_errorI_C,statistics_fidelity_C]
#save = statistics_save("stats_1-23.dat",RBM_stats=RBM_stats_)  
#save = statistics_save_GNUPLOT("stats_1-10_gnuplot.dat",RBM_stats=RBM_stats_)  


#A = Read_RBM("stats_11-20.dat",Stats="stats")
#%%
plt.subplot(311)
plt.errorbar(Omega,statistics_errorE_C[2], np.sqrt(statistics_errorE_C[3]),barsabove=True, ls="-",marker=".",label="test")
plt.plot(Omega,statistics_errorE_C[2], ".-")
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
plt.errorbar(Omega,statistics_fidelity_C[2], np.sqrt(statistics_fidelity_C[3]),barsabove=True, ls="-",marker=".",label="test")
#plt.fill_between(Omega, statistics_fidelity[1][0], statistics_fidelity[1][1], color='#539ecd')
plt.plot(Omega,statistics_fidelity_C[2], ".-")
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
plt.plot(Omega,statistics_errorI_C[2], ".-")
plt.plot(Omega,statistics_errorF_C[2], ".-")
plt.errorbar(Omega,statistics_errorI_C[2], np.sqrt(statistics_errorI_C[3]),barsabove=True, ls="-",marker=".",label="test")
plt.errorbar(Omega,statistics_errorF_C[2], np.sqrt(statistics_errorF_C[3]),barsabove=True, ls="-",marker=".",label="test")

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
#iteration1,fidelity_list1,loss_list1,RBM_Floquet_spectrum_up1,RBM_Floquet_spectrum_down1,Floquet_spectrum_up1,Floquet_spectrum_down1 = RBM_DataAnalysis('RBM_TrainingFloquet_WaveFunction_9stRun.dat')
#iteration2,fidelity_list2,loss_list2,RBM_Floquet_spectrum_up2,RBM_Floquet_spectrum_down2,Floquet_spectrum_up2,Floquet_spectrum_down2 = RBM_DataAnalysis('RBM_TrainingFloquet_WaveFunction_8stRun.dat')

#%%
Omega = np.linspace(1,N,N)-1#(10.0/N)*(np.linspace(1,N,N)-1)
plt.plot(Omega,np.abs(-RBM_Floquet_spectrum_up+RBM_Floquet_spectrum_down), ".-")
#plt.plot(Omega,np.abs(RBM_Floquet_spectrum_up1-RBM_Floquet_spectrum_down1), ".-")
#plt.plot(Omega,np.abs(RBM_Floquet_spectrum_up2-RBM_Floquet_spectrum_down2), ".-")
plt.plot(Omega,Floquet_spectrum_up-Floquet_spectrum_down, ".-")
#plt.plot(Omega,Floquet_spectrum_up-Floquet_spectrum_down+2.7*1, ".-")
#plt.plot(Omega,Floquet_spectrum_up-Floquet_spectrum_down+2.7*2, ".-")
#plt.plot(Omega,Floquet_spectrum_up-Floquet_spectrum_down+2.7*3, ".-")
#plt.plot(Omega,Floquet_spectrum_up-Floquet_spectrum_down+2.7*4, ".-")
#plt.plot(Omega,Floquet_spectrum_up-Floquet_spectrum_down+2.7*5, ".-")
#plt.plot(Omega,Floquet_spectrum_up-Floquet_spectrum_down+2.7*9, ".-")
#plt.yscale('log')
plt.xlabel("Driving amplitude")
plt.ylabel("Qubit gap")
plt.show()
#%%
data_ = np.zeros([32,5],dtype=np.double)
data_[:,0] = Omega
data_[:,1] = np.abs(RBM_Floquet_spectrum_up-RBM_Floquet_spectrum_down)
data_[:,2] = np.abs(RBM_Floquet_spectrum_up1-RBM_Floquet_spectrum_down1)
data_[:,3] = np.abs(RBM_Floquet_spectrum_up2-RBM_Floquet_spectrum_down2)
data_[:,4] = np.abs(Floquet_spectrum_up-Floquet_spectrum_down)

#file_data = open('QubitGap.dat','w')
#file_data.write('#Omega RBM RBM1 RBM2 Floquet')
#np.savetxt(file_data,data_)


