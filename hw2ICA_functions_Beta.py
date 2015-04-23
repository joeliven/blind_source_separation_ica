import scipy as sc
import scipy.io as scio
import scipy.io.wavfile as wav
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
import scipy.linalg as lin
import scipy.stats as st
import math
import os
from xlwt import Row
from astropy.table import row

# STEP 3: Calculate Y = WX ... Y is our current estimate of the source signal matrix, U:
def calcY(W, X):
    W = np.asmatrix(W)
    X = np.asmatrix(X)
    Y = W*X
    return Y

# Calculate the total error between our current estimation, Y, and the actual source signals, U: 
def calcError_Norm(U, Y):
    U = np.asmatrix(U)
    Y = np.asmatrix(Y)
    error = lin.norm(U) - lin.norm(Y)
    return error    

def calcError_Correlation(U, Y):
    U_ = np.array(U)
    Y_ = np.array(Y)
    
    corrcoefs_all = list()
    for i in xrange(0, len(U_[:,0])):
        U_row = U_[i,:]
        corrcoefs_perSig = list()
        for ii in xrange(0, len(Y_[:,0])):
            Y_row = Y_[ii,:]
            cf = (np.corrcoef(U_row, Y_row))[1,0]
            corrcoefs_perSig.append(abs(cf))
            
        corrcoefs_perSig = sorted(corrcoefs_perSig, reverse=True)
        corrcoefs_all.append(corrcoefs_perSig[0])
        
    corrcoefs = np.array(corrcoefs_all)
    avg_corrcoef = np.mean(corrcoefs)
    error = 1 - avg_corrcoef
    return error, corrcoefs   

def calcZ(Y_, b):
    Y = np.array(Y_)
    Z = np.empty_like(Y)
    for i in xrange(0, len(Y[:,0])):
        for j in xrange(0, len(Y[0,:])):
            y =  Y[i,j]
#             print("y is: " + str(y))
            try:
                temp = math.exp(-1*y*(b[i,0]))
                z = 1.0/(1.0 + temp)
            except OverflowError:
                if(y < 0):
                    print(".......................OverflowError handled........y < 0......................")
                    z = 0
                else:
                    print(".......................OverflowError handled........y >= 0......................")
                    z = 1
            Z[i,j] = z
    Z = np.asmatrix(Z)    
    return Z

def calcDeltaW(n, t, lr, Z, Y, W):
    I = np.identity(n, dtype=np.float)
    
    I = np.asmatrix(I)
    Z = np.asmatrix(Z)
    Y = np.asmatrix(Y)
    W = np.asmatrix(W)
    
    delta_W = lr*((I+((1-(2*Z))*(Y.T)))*W)
    delta_W = np.asmatrix(delta_W)
    return delta_W


def createWAVfiles(M, exp_ID, file_prefix, sample_rate):
    file_suffix = ".wav"
    exp_prefix = "exp" + str(exp_ID) + "_"
    filename = ""
    K = np.array(M)
    K = scale_min_max(K)
    for i in xrange(0, len(K[:,0])):
        filename = exp_prefix + file_prefix + str(i) + file_suffix
        wav.write(filename, sample_rate, K[i,:])

def calcLR(lr_start, lr_end, lr_anneal_type, lr_dcy_bs, lr_dcy_rt, max_itrs, it_num):
    if(lr_anneal_type == "none"):
        return lr_start
    if(lr_anneal_type == "linear"):
        slope = float(lr_end - lr_start)/float(max_itrs)
        lr = lr_start + (slope * it_num)
        return lr
    if(lr_anneal_type == "decay"):
        return ((lr_start - lr_end) * math.pow(lr_dcy_bs, math.pow(it_num, lr_dcy_rt))) + lr_end
        
def show_src_sigs_orig(U, expID, plot_start, plot_end):
    U_temp = np.array(U)
    U_temp = scale_min_max(U_temp)
    fig, ax = plt.subplots()
    title = "exp" + str(expID) + "_src_sigs_orig"
    fig.suptitle(title, fontsize='20')
    if(plot_start == 0):
        for row in xrange(0, len(U_temp[:,0])):
            signal = U_temp[row,:]
            signal += row
            ax.plot(signal, label="src_sig_orig_" + str(row))
    else:
        for row in xrange(0, len(U_temp[:,0])):
            signal = U_temp[row,plot_start:plot_end]
            signal += row
            ax.plot(signal, label="src_sig_orig_" + str(row))
    plt.xlabel('time, t', fontsize=16)
    plt.ylabel('signal magnitude', fontsize=16)
    legend = ax.legend(loc='best', shadow=True)
    plt.savefig(title)
#     plt.show()
#     plt.close()
    
def show_mixed_sigs(X, expID, plot_start, plot_end):
    X_temp = np.array(X)
    X_temp = scale_min_max(X_temp)
    fig, ax = plt.subplots()
    title = "exp" + str(expID) + "_mixed_sigs"
    fig.suptitle(title, fontsize='20')
    if(plot_start == 0):
        for row in xrange(0, len(X_temp[:,0])):
            signal = X_temp[row,:]
            signal += row
            ax.plot(signal, label="mixed_sig_" + str(row))
    else:
         for row in xrange(0, len(X_temp[:,0])):
            signal = X_temp[row,plot_start:plot_end]
            signal += row
            ax.plot(signal, label="mixed_sig_" + str(row))       
        
    plt.xlabel('time, t', fontsize=16)
    plt.ylabel('signal magnitude', fontsize=16)
    legend = ax.legend(loc='best', shadow=True)
    plt.savefig(title)
#     plt.show()
#     plt.close()
    
def show_src_sigs_comp(U, Y, expID, path, it_num, plot_start, plot_end):
    U_temp = np.array(U)
    U_temp = scale_min_max(U_temp)
    Y_temp = np.array(Y)
    Y_temp = scale_min_max(Y_temp)
    fig, ax = plt.subplots()
    title = "exp" + str(expID) + "_src_sigs_comp_itNum_" + str(it_num)
    fig.suptitle(title, fontsize='20')
    if(plot_start == 0):
        for row in xrange(0, len(U_temp[:,0])):
            signal = U_temp[row,:]
            signal += 2*row
            ax.plot(signal, label="src_sig_orig_" + str(row))
        for row in xrange(0, len(Y_temp[:,0])):
            signal = Y_temp[row,:]
            signal += 2*row
            signal += 1
            ax.plot(signal, label="src_sig_recon_" + str(row))
    else:
        for row in xrange(0, len(U_temp[:,0])):
            signal = U_temp[row,plot_start:plot_end]
            signal += 2*row
            ax.plot(signal, label="src_sig_orig_" + str(row))
        for row in xrange(0, len(Y_temp[:,0])):
            signal = Y_temp[row,plot_start:plot_end]
            signal += 2*row
            signal += 1
            ax.plot(signal, label="src_sig_recon_" + str(row))
                    
    plt.xlabel('time, t', fontsize=16)
    plt.ylabel('signal magnitude', fontsize=16)
    legend = ax.legend(loc='best', shadow=True)
    os.chdir(path)
    plt.savefig(title)
    plt.show()

def scale_min_max(M):
    K = np.array(M)
    K_scaled = np.empty_like(K)
    for i in xrange(0, len(K[:,0])):
        row = K[i,:]
        max = np.amax(row)
        min = np.amin(row)
        scaled_row = (row - min)/(max - min)
        K_scaled[i,:] = scaled_row
    return K_scaled


