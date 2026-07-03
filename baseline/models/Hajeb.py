import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.svm import SVC
import h5py
import cv2
import pandas as pd
import os
import time
from datetime import datetime
import argparse

def binary(x,y):
    data = []
    label = []  
    t0 = np.where(y == 0)[0]
    t4 = np.where(y == 4)[0]
    
    for i in range(len(x)):
        if i in t0:
            data.append(x[i])
            label.append(y[i])
    
        if i in t4:
            data.append(x[i])
            label.append(y[i])
    return np.array(data), np.array(label)


def split(X, Y):
    print(set(Y))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify = Y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify = y_train)

    weight = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    d_class_weights = dict(enumerate(weight))


    return X_train, X_val, X_test, y_train, y_val, y_test, d_class_weights


# https://github.com/tzm030329/GLCM/blob/master/fast_glcm.py
def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=3):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm

def fast_glcm_mean(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm mean
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    return mean

def fast_glcm_std(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm std
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    return std
    
def fast_glcm_var(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm variance
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i,j] * i - mean)**2

    #std = np.sqrt(std2)
    return std2

def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i,j] * (i-j)**2

    return cont

def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i,j] * np.abs(i-j)

    return diss

def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo

def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene

def fast_glcm_max(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    max_  = np.max(glcm, axis=(0,1))
    return max_

def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent

def glcm_difference(X):
    glcm_dif = []

    nbit = 8
    ks = 3
    mi, ma = 0, 1

    for i in range(len(X)):

        img = X[i][0]
        h,w = img.shape

        img[:,:w//2] = img[:,:w//2]//2+127
        mean_pre = fast_glcm_mean(img.copy(), mi, ma, nbit, ks)
        var_pre = fast_glcm_var(img.copy(), mi, ma, nbit, ks)

        img = X[i][1]
        h,w = img.shape

        img[:,:w//2] = img[:,:w//2]//2+127
        mean_post = fast_glcm_mean(img.copy(), mi, ma, nbit, ks)
        var_post = fast_glcm_var(img.copy(), mi, ma, nbit, ks)
        

        mean_diff = mean_pre.flatten() - mean_post.flatten()
        var_diff = var_pre.flatten() - var_post.flatten()
        
        #diff = np.concatenate((mean_diff, var_diff), axis=0)
        #diff = mean_diff + var_diff
        
        glcm_dif.append(var_diff)

    return np.array(glcm_dif)

def dsm_difference(X):
    dsm_dif = []
    for i in range(len(X)):

        pre = X[i][0]

        post = X[i][1]

        dsm_dif.append(pre.flatten() - post.flatten())

    return np.array(dsm_dif)

if __name__ == "__main__":
    tf.random.set_seed(1234)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description="SVM Experiment")
    parser.add_argument("--experiment", type=int, required=True, help="Experiment number")
    args = parser.parse_args()

    h5f = h5py.File('Data/3blur_all_data16x16.h5', 'r')
    X_train = h5f['X_train'][:]
    y_train = h5f['Y_train'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['Y_val'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['Y_test'][:]
    h5f.close()

    if args.experiment == 1:
        X_train, y_train = binary(X_train, y_train)
        X_val, y_val = binary(X_val, y_val)
        X_test, y_test = binary(X_test, y_test)
        y_train[y_train == 4] = 1
        y_test[y_test == 4] = 1
        y_val[y_val == 4] = 1

    elif args.experiment == 2:
        y_train[y_train == 0] = 0
        y_train[y_train == 1] = 0
        y_train[y_train == 2] = 0
        y_train[y_train == 3] = 0
        y_train[y_train == 4] = 1

        y_test[y_test == 0] = 0
        y_test[y_test == 1] = 0
        y_test[y_test == 2] = 0
        y_test[y_test == 3] = 0
        y_test[y_test == 4] = 1

        y_val[y_val == 0] = 0
        y_val[y_val == 1] = 0
        y_val[y_val == 2] = 0
        y_val[y_val == 3] = 0
        y_val[y_val == 4] = 1
    else:
        pass

    extracted_X_train = dsm_difference(X_train.copy())
    extracted_X_val = dsm_difference(X_val.copy())
    extracted_X_test = dsm_difference(X_test.copy())
    print("woi")
    del X_train
    del X_val
    del X_test

    weight = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    d_class_weights = dict(enumerate(weight))

    # Param

    kernel = ['linear', 'poly', 'rbf', 'sigmoid'] #'precomputed'
    dec_func = ['ovo', 'ovr']
    slack = [0.1, 1, 10]
    gamma = ['auto', 'scale', 1, 0.1, 0.01, 0.001, 0.0001]
    fm_ = -99999
    temp = 0

    # Define the CSV file name
    csv_filename = f"hajeb_experiment_{args.experiment}.csv"

    # Initialize a list to store all results
    all_results = []

    for func in dec_func:
        for i in kernel:
            for c in slack:
                for g in gamma:
                    clf = SVC(decision_function_shape=func, kernel=i, C=c, gamma=g, class_weight = d_class_weights)

                    clf.fit(extracted_X_train, y_train)
                    prediction = clf.predict(extracted_X_test)
                    if args.experiment == 1 or args.experiment == 2:
                        acc = accuracy_score(y_test, prediction)
                        prec = precision_score(y_test, prediction)
                        rec = recall_score(y_test, prediction)
                        fm = f1_score(y_test, prediction)
                        confus = confusion_matrix(y_test, prediction)
                    else:
                        acc = accuracy_score(y_test, prediction)
                        prec = precision_score(y_test, prediction, average='macro')
                        rec = recall_score(y_test, prediction, average='macro')
                        fm = f1_score(y_test, prediction, average='macro')
                        confus = confusion_matrix(y_test, prediction)

                    print("Decision Function: ", func)
                    print("Kernel: ", i)
                    print("Slack: ", c)
                    print("Gamma: ", g)

                    print("F-Measure: ", fm)
                    print("Accuracy: ", acc)
                    print("F-Measure: ",fm)
                    print("Precision: ",prec)
                    print("Recall: ",rec)
                    print(confus)
                    # fm = np.mean(fm)
                    if(fm_ < fm):
                        best_func = func
                        best_kernel = i
                        best_slack = c
                        best_gamma = g
                        best_model = clf
                        best_confus = confus
                        fm_ = fm

                    # Store results
                    all_results.append([func, i, c, g, acc, fm, prec, rec, confus])

                    temp = temp + 1
                    print(temp)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results, columns=["Decision Function", "Kernel", "Slack", "Gamma", "Accuracy", "F-Measure", "Precision", "Recall", "Confusion matrix"])

    # Save results to CSV
    results_df.to_csv(csv_filename, index=False)

    print("Grid search results saved to", csv_filename)

    print("Decision Function: ", best_func)
    print("Kernel: ", best_kernel)
    print("Slack: ", best_slack)
    print("Gamma: ", best_gamma)
    print("F-Measure: ", fm_)

    # Define the directory path
    save_dir = "saved_model/Hajeb/"

    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    with open('saved_model/Hajeb/'+str(fm_)+'_func_'+best_func+"_kernel_"+best_kernel+"_slack_"+str(best_slack)+"_gamma_"+str(best_gamma)+"_date_"+str(datetime.now())+'.pkl', 'wb') as outp:
        pickle.dump(best_model, outp, pickle.HIGHEST_PROTOCOL)

    # 2. Save classification results for the best model
    results_dir = "results/Hajeb/"
    os.makedirs(results_dir, exist_ok=True)

    # Get predictions from best model
    y_pred = best_model.predict(extracted_X_test)
    y_prob = best_model.decision_function(extracted_X_test)  # For SVM, this gives decision values

    # Get decision scores (SVM) or probabilities (LogisticRegression, etc.)
    if hasattr(best_model, "decision_function"):
        y_scores = best_model.decision_function(extracted_X_test)
    elif hasattr(best_model, "predict_proba"):
        y_scores = best_model.predict_proba(extracted_X_test)
    else:
        raise ValueError("Model does not support decision_function or predict_proba")

    # Reshape for binary case (1D -> 2D for consistency)
    if len(y_scores.shape) == 1:
        y_scores = y_scores.reshape(-1, 1)

    # Create results DataFrame
    results_dict = {
        'true_label': y_test,
        'predicted_label': y_pred,
    }

    # Add decision scores/probabilities (handles multiclass)
    n_classes = y_scores.shape[1]
    if n_classes == 1:
        results_dict['decision_value'] = y_scores.flatten()
    elif n_classes == 2:  # Binary case (store only positive class score)
        results_dict['decision_value'] = y_scores[:, 1]
    else:  # Multiclass: store scores for each class
        for i in range(n_classes):
            results_dict[f'decision_value_class_{i}'] = y_scores[:, i]

    results_dict = pd.DataFrame(results_dict)

    # Save results to CSV
    results_filename = f"{results_dir}Hajeb_best_model_results_experiment_{args.experiment}.csv"
    results_dict.to_csv(results_filename, index=False)
    print(f"Saved classification results to: {results_filename}")
