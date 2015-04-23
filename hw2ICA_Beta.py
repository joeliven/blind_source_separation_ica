import scipy as sc
import scipy.io as scio
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
import hw2ICA_functions_Beta as fn
import scipy.linalg as lin
import os
import math

# CONSTANTS:
max_possible_src_sigs = 5
sample_rate = 11025
src_sig_orig = "src_orig_sig"
mixed_sig = "mixed_sig"
src_sig_reconstructed = "src_reconstructed_sig"

raw_data = list()
A = list()
U_all = list()
max_t = 0

# User Input Arguments, specifying details of experiment run:
test = raw_input("Real Data or Test Data? ('r' or 't') ")
hw2_exp_id = raw_input("experiment ID: ")
print("which source signals to use:")
which_src_sigs = list()
for num in xrange(0, max_possible_src_sigs):
    if((raw_input("\tsrc sig " + str(num + 1) + "? (y/n) ")) == "y"):
        which_src_sigs.append(num)
src_sigs = np.asarray(which_src_sigs)
n = len(src_sigs)
m = int(raw_input("number of mixed signals: (greater than n): "))

if(test == "t"):
    test_data_set = scio.loadmat("icaTest.mat")
    raw_data = test_data_set["U"]
    U_all = np.array(raw_data)
    max_t = len(U_all[0,:])
    A = test_data_set["A"]
    A = np.matrix(A)
    print("A is:" )
    print(A)
else:    
    # Get raw data:
    raw_data = scio.loadmat("sounds.mat")["sounds"]
    U_all = np.array(raw_data)
    max_t = len(U_all[0,:])
    # generate random mixing matrix A:
    A = np.random.random((m, n))
#     A = (.1 * np.random.random((m, n)))
#     A = (.01 * np.random.random((m, n)))
#     A = [.10665277, .77491046, .08443588, .96189808, .81730322, .39978265, .00463422, .86869471, .2598704]
#     A = np.matrix(A)
#     A = np.reshape(A, (3,3))

t_temp = (raw_input("how much of each signal, t: (0, 44,000] or 'all': "))
if t_temp == "all":
    t = max_t
else:
    t = int(t_temp)
lr_start = float(raw_input("starting learning rate, lr_start: (0,1) "))
# ending learning rate, lr_end (if using annealing):
lr_end = float(raw_input("ending learning rate, lr_start: (0,1) "))
if(lr_start != lr_end):
    lr_anneal_type = raw_input("learning rate annealing type: (linear or decay) ")
    if(lr_anneal_type == "decay"):
        # set the base value for the exponential decay of the learning rate:
        lr_dcy_bs = float(raw_input("learning rate exponential decay base value: [.8, 1) "))
        # set the root value for the exponential decay of the learning rate:
        lr_dcy_rt = float(raw_input("learning rate exponential decay kth root value\n(taken as exponent...e.g. for the fourth root of t, enter .25): (0, 1) "))
    else:
        lr_dcy_bs = "na"
        lr_dcy_rt = "na"
else:
    lr_anneal_type = "none"
    lr_dcy_bs = "na"
    lr_dcy_rt = "na"
add_header = raw_input("add header? (y/n) ") 
convg_lim = float(raw_input("convg limit (Frobenius norm of delta_W): (0,???) "))
max_itrs = int(raw_input("max iterations: "))
print_every = int(raw_input("print_every: "))
write_to_file_every = int(raw_input("write_to_file_every: "))
show_src_sigs_comp_every = int(raw_input("show_src_sigs_comp_every: "))
plot_zoom = raw_input("use how many time slices for displaying plots? ")
plot_start = 0
plot_end = 0
if(plot_zoom != "all"):
    plot_zoom = int(plot_zoom)
    plot_start = np.random.randint(1000, (max_t - plot_zoom - 1000))
    plot_end = plot_start + plot_zoom
print("plot_start=" + str(plot_start))
print("plot_end=" + str(plot_end))

try:
    # create file in subdirectory for output of metadata for each experiment:
    fname = "hw2_exp" + str(hw2_exp_id) + ".csv"
    cur_dir = os.getcwd()
    path_name = "exp" + str(hw2_exp_id)
    path = os.path.join(cur_dir, path_name)
    if os.access(path, os.F_OK):
        os.chdir(path)
        out = open(fname, 'w')
    else:
        os.mkdir(path)
        os.chdir(path)
        out = open(fname, 'w')
    
    # write experiment parameters out to file:
    out_header = ("expID,"
        "src_sigs (0-indx),"
        "n,"
        "m,"
        "t,"
        "lr_start,"
        "lr_end,"
        "lr_anneal_type,"
        "lr_dcy_bs,"
        "lr_dcy_rt,"
        "convg_lim,"
        "max_itr\n")
    out.write(out_header)
    out_header_data = str(hw2_exp_id) +","\
        + str(src_sigs) +","\
        + str(n) +","\
        + str(m) +","\
        + str(t) +","\
        + str(lr_start) +","\
        + str(lr_end) +","\
        + lr_anneal_type +","\
        + str(lr_dcy_bs) +","\
        + str(lr_dcy_rt) +","\
        + str(convg_lim) +","\
        + str(max_itrs)
    out.write(out_header_data)
    out_sub_header = ("\n\nit_num,"
        "lr,"
        "convg,"
        "err_cor,"
        "err_cor_per_sig,"
        "err_norm\n")
    out.write(out_sub_header)
    
    # use the user-specified n rows of raw data to create U:
    U = U_all[src_sigs,0:t]
    U = np.matrix(U)
    
    # generate original source signal wav files:
    fn.createWAVfiles(U, "", src_sig_orig, sample_rate)
    # generate plots for original source signal wav files:
    fn.show_src_sigs_orig(U, hw2_exp_id, plot_start, plot_end)
    U = np.asmatrix(U)
    
    # STEP 1: Assume X = AU
    X = A*U
    fn.show_mixed_sigs(X, hw2_exp_id, plot_start, plot_end)
    X = np.matrix(X)
    
    # generate mixed signal wav files:
    fn.createWAVfiles(X, hw2_exp_id, mixed_sig, sample_rate)
    
    # STEP 2: Initialize W with small random values:
    W = (.1 * np.random.random((n,m)))
#     W = (.05 * np.random.random((n,m)))
#     W = (.01 * np.random.random((n,m)))
    W = np.matrix(W)
    b = np.random.random((n,1))
     
    error_cor = -1
    error_norm = -1
    Y = np.empty_like(U)
    lr = 1.0
    convg = 100000000.0
    it_num = -1    
    
    while(convg > convg_lim and it_num < max_itrs): 
        if(it_num == -1):
            # STEP 3: Calculate Y = WX ... Y is our current estimate of the source signal matrix, U:
            Y = W*X

            # Calculate the initial error measurement between Y and U:
            error_cor, corrcoefs = fn.calcError_Correlation(U, Y)
            error_norm = fn.calcError_Norm(U,Y)
            print("it_num=" + str(it_num) + "  lr=" + str(lr_start) + "  covg=N/A" + "  init_err_cor=" + str(error_cor) + "  init_err_norm=" + str(error_norm))
            out_string = str(it_num) +","\
                + str(lr_start) +","\
                + "" +","\
                + str(error_cor) +","\
                + str(error_norm) + "\n"
            out.write(out_string)
            it_num += 1
    
        # STEP 4: Calculate Z where z i,j  = g(y i,j)
        Z = fn.calcZ(Y, b)
        
        # STEP 5: Find Delta(W) = lr_start((I + (1 -2Z)Ytranspose)W
        lr = fn.calcLR(lr_start, lr_end, lr_anneal_type, lr_dcy_bs, lr_dcy_rt, max_itrs, it_num)
        delta_W = fn.calcDeltaW(n, t, lr, Z, Y, W)
            
        # STEP 6: update W = W + delta_W
        W = W + delta_W
        convg = lin.norm(delta_W)
        
        # STEP 3: (moving this to the end of the loop...same effect though) 
        # Calculate Y = WX ... Y is our current estimate of the source signal matrix, U:
        W = np.asmatrix(W)
        X = np.asmatrix(X)
        Y = W*X
        
        # Calculate the current error measurement between Y and U:
        if((it_num % write_to_file_every == 0) or (it_num % print_every == 0)):
            error_cor, corrcoefs = fn.calcError_Correlation(U, Y)
            error_norm = fn.calcError_Norm(U,Y)            
        if(it_num % write_to_file_every == 0):
            out_string = str(it_num) +","\
                + str(lr) +","\
                + str(convg) +","\
                + str(error_cor) +","\
                + str(corrcoefs) +","\
                + str(error_norm) + "\n"
            out.write(out_string)
        if(it_num % print_every == 0):
            print("it_num=" + str(it_num) + "  lr=" + str(round(lr, 5)) + "  covg=" + str(round(convg, 10)) + "  err_cor=" + str(round(error_cor, 6)) + "  err_norm=" + str(round(error_norm, 6)))
        if(show_src_sigs_comp_every != 0 and it_num % show_src_sigs_comp_every == 0):
            fn.show_src_sigs_comp(U, Y, hw2_exp_id, path, it_num, plot_start, plot_end)
        it_num += 1
    
    # calculate the final error:
    error_cor, corrcoefs = fn.calcError_Correlation(U, Y)
    error_norm = fn.calcError_Norm(U,Y)
    print("FINAL CONVERGENCE: " + str(convg))
    print("FINAL ERR_COR: = " + str(error_cor))
    print("FINAL ERR_NORM: = " + str(error_norm))

    # generate reconstructed source signal wav files:
    fn.createWAVfiles(Y, hw2_exp_id, src_sig_reconstructed, sample_rate)
    print(W)

finally:
    out.close()

try:
    # create file to track final results of all experiments:
    os.chdir(os.path.pardir)
    f_results = open("hw2_results.csv", 'a')
    if add_header == 'y':
        results_header = ("expID,"
        "final_error_cor,"
        "final_error_norm,"
        "src_sigs (0-indx),"
        "n,"
        "m,"
        "t,"
        "lr_start,"
        "lr_end,"
        "lr_anneal_type,"
        "lr_dcy_bs,"
        "lr_dcy_rt,"
        "convg_lim,"
        "max_itr\n")
        
        f_results.write(results_header)
        
    # output summary of results to master results file:
    results = str(hw2_exp_id) +","\
        + str(error_cor) +","\
        + str(error_norm) +","\
        + str(src_sigs) +","\
        + str(n) +","\
        + str(m) +","\
        + str(t) +","\
        + str(lr_start) +","\
        + str(lr_end) +","\
        + lr_anneal_type +","\
        + str(lr_dcy_bs) +","\
        + str(lr_dcy_rt) +","\
        + str(convg_lim) +","\
        + str(max_itrs) + "\n"
    
    f_results.write(results)
#     fn.show_src_sigs_orig(U, hw2_exp_id)
#     fn.show_mixed_sigs(X, hw2_exp_id)
    fn.show_src_sigs_comp(U, Y, hw2_exp_id, path, it_num, plot_start, plot_end)
finally:
    f_results.close()    
