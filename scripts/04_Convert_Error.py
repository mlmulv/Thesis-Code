import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join("..", "src")))

def main():
    try:
        SI_errs_integral_static_mse = np.load("saved_variables/module4/SI_errs_integral_static_mse_04.npy")
        SI_errs_integral_vary_mse = np.load("saved_variables/module4/SI_errs_integral_vary_mse_04.npy")
        SI_errs_augment_static_mse = np.load("saved_variables/module4/SI_errs_augment_static_mse_04.npy")
        SI_errs_augment_vary_mse = np.load("saved_variables/module4/SI_errs_augment_vary_mse_04.npy")
        print("Previous run variables already exist")

    except Exception:
        SI_ests_integral_static = np.load("saved_variables/module4/SI_ests_integral_static_04.npy")
        SI_ests_integral_vary = np.load("saved_variables/module4/SI_ests_integral_vary_04.npy")
        SI_ests_augment_static = np.load("saved_variables/module4/SI_ests_augment_static_04.npy")
        SI_ests_augment_vary = np.load("saved_variables/module4/SI_ests_augment_vary_04.npy")
        SI_true_arr_integral_static = np.load("saved_variables/module4/SI_true_arr_integral_static_04.npy")
        SI_true_arr_integral_vary = np.load("saved_variables/module4/SI_true_arr_integral_vary_04.npy")
        SI_true_arr_augment_static = np.load("saved_variables/module4/SI_true_arr_augment_static_04.npy")
        SI_true_arr_augment_vary = np.load("saved_variables/module4/SI_true_arr_augment_vary_04.npy")

        SI_errs_integral_static_mse = np.zeros_like(SI_ests_integral_static)
        SI_errs_integral_vary_mse = np.zeros_like(SI_ests_integral_vary)
        SI_errs_augment_static_mse = np.zeros_like(SI_ests_augment_static)
        SI_errs_augment_vary_mse = np.zeros_like(SI_ests_augment_vary) 
    
        for i in range(SI_true_arr_integral_static.shape[0]):
            SI_true = SI_true_arr_integral_static[i,:]
            SI_est = SI_ests_integral_static[i,:]
            SI_errs_integral_static_mse[i,:] = (SI_true - SI_est)**2

        for i in range(SI_true_arr_integral_vary.shape[0]):
            SI_true = SI_true_arr_integral_vary[i,:]
            SI_est = SI_ests_integral_vary[i,:]
            SI_errs_integral_vary_mse[i,:] = (SI_true - SI_est)**2

        for i in range(SI_true_arr_augment_static.shape[0]):
            SI_true = SI_true_arr_augment_static[i,:]
            for j in range(SI_ests_augment_static.shape[1]):
                for k in range(SI_ests_augment_static.shape[0]):
                    SI_est = SI_ests_augment_static[k,j,i,:]
                    SI_errs_augment_static_mse[k,j,i,:] = (SI_true - SI_est)**2


        for i in range(SI_true_arr_augment_vary.shape[0]):
            SI_true = SI_true_arr_augment_vary[i,:]
            for j in range(SI_ests_augment_vary.shape[1]):
                for k in range(SI_ests_augment_vary.shape[0]):
                    SI_est = SI_ests_augment_vary[k,j,i,:]
                    SI_errs_augment_vary_mse[k,j,i,:] = (SI_true - SI_est)**2

        np.save("saved_variables/module4/SI_errs_integral_static_mse_04", SI_errs_integral_static_mse)
        np.save("saved_variables/module4/SI_errs_integral_vary_mse_04", SI_errs_integral_vary_mse)
        np.save("saved_variables/module4/SI_errs_augment_static_mse_04", SI_errs_augment_static_mse)
        np.save("saved_variables/module4/SI_errs_augment_vary_mse_04", SI_errs_augment_vary_mse)

if __name__ == "__main__":    
    main()