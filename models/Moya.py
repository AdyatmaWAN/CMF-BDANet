import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.svm import SVC
import h5py
import csv
import os
import argparse

tf.random.set_seed(1234)
np.random.seed(0)
import time
from datetime import datetime

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



def feature_difference(X):
    dsm_dif = []
    for i in range(len(X)):

        pre = X[i][0]

        post = X[i][1]

        diff = pre.flatten() - post.flatten()
        avg_diff = np.mean(diff)
        std = np.std(diff)
        r =  np.corrcoef(pre.flatten(), post.flatten())[0,1]

        #std = np.sqrt(sum((diff-avg_diff)**2) / len(diff))
        #upper_side = (len(diff) * sum(pre.flatten() * post.flatten())) - (sum(pre.flatten()) * sum(post.flatten()))
        #sum(pre.flatten()**2) (sum(pre.flatten())**2)
        
        dsm_dif.append(np.array((avg_diff, std, r)))

    return np.array(dsm_dif)

# h5f = h5py.File('Data/3blur_all_data16x16.h5', 'r')
# X_train = h5f['X_train'][:]
# y_train = h5f['Y_train'][:]
# X_val = h5f['X_val'][:]
# y_val = h5f['Y_val'][:]
# X_test = h5f['X_test'][:]
# y_test = h5f['Y_test'][:]
# h5f.close()
#
#
# #X_train, y_train = binary(X_train, y_train)
# #X_val, y_val = binary(X_val, y_val)
# #X_test, y_test = binary(X_test, y_test)
#
# '''
# y_train[y_train == 0] = 0
# y_train[y_train == 1] = 0
# y_train[y_train == 2] = 0
# y_train[y_train == 3] = 0
# y_train[y_train == 4] = 1
#
# y_test[y_test == 0] = 0
# y_test[y_test == 1] = 0
# y_test[y_test == 2] = 0
# y_test[y_test == 3] = 0
# y_test[y_test == 4] = 1
#
# y_val[y_val == 0] = 0
# y_val[y_val == 1] = 0
# y_val[y_val == 2] = 0
# y_val[y_val == 3] = 0
# y_val[y_val == 4] = 1
# '''
#
# extracted_X_train = feature_difference(X_train.copy())
# print(extracted_X_train.shape)
# extracted_X_val = feature_difference(X_val.copy())
# extracted_X_test = feature_difference(X_test.copy())
#
# del X_train
# del X_val
# del X_test
#
# # Param
# #Grid
# kernel = ['linear', 'poly', 'rbf', 'sigmoid']
# dec_func = ['ovo', 'ovr']
# slack = [0.1, 1, 10, 100]
# gamma = ['auto', 'scale', 1, 0.1, 0.01, 0.001, 0.0001]
#
# weight = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
# d_class_weights = dict(enumerate(weight))
#
# fm_ = -99999
# temp = 0
# for func in dec_func:
#     for i in kernel:
#         for c in slack:
#             for g in gamma:
#                 clf = SVC(decision_function_shape=func, kernel=i, C=c, gamma=g, class_weight = d_class_weights)
#
#                 clf.fit(extracted_X_train, y_train)
#                 prediction = clf.predict(extracted_X_test)
#                 fm = f1_score(y_test, prediction, average = None)
#
#                 print("Decision Function: ", func)
#                 print("Kernel: ", i)
#                 print("Slack: ", c)
#                 print("Gamma: ", g)
#
#                 acc = accuracy_score(y_test, prediction)
#                 prec = precision_score(y_test, prediction, average = None)
#                 rec = recall_score(y_test, prediction, average = None)
#                 confus = confusion_matrix(y_test, prediction)
#                 fm = np.mean(fm)
#                 print("F-Measure: ", fm)
#                 print("Accuracy: ", acc)
#                 print("F-Measure: ",fm)
#                 print("Precision: ",prec)
#                 print("Recall: ",rec)
#                 print(confus)
#                 if(fm_ < fm):
#                     best_func = func
#                     best_kernel = i
#                     best_slack = c
#                     best_gamma = g
#                     best_model = clf
#                     fm_ = fm
#                 temp = temp + 1
#                 print(temp)
#
#
# print("Decision Function: ", best_func)
# print("Kernel: ", best_kernel)
# print("Slack: ", best_slack)
# print("Gamma: ", best_gamma)
# print("F-Measure: ", fm_)
# with open('save_model/Moya/'+str(fm_)+'_func_'+best_func+"_kernel_"+best_kernel+"_slack_"+str(best_slack)+"_gamma_"+str(best_gamma)+"_date_"+str(datetime.now())+'.pkl', 'wb') as outp:
#     pickle.dump(best_model, outp, pickle.HIGHEST_PROTOCOL)


# acc = accuracy_score(y_test, predicted)
#
# fm = f1_score(y_test, predicted, average=None)
# acc = accuracy_score(y_test, predicted)
# prec = precision_score(y_test, predicted, average=None)
# rec = recall_score(y_test, predicted, average=None)
# confus = confusion_matrix(y_test, predicted)
#
# print("----------------------------------------------------------------------------")
# print("Accuracy: ", acc)
# print("F-Measure: ",fm)
# print("Precision: ",prec)
# print("Recall: ",rec)
# print(confus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Experiment")
    parser.add_argument("--experiment", type=int, required=True, help="Experiment number")
    args = parser.parse_args()

    # Create CSV file and write the headers
    csv_filename = f"csv/moya_experiment_{args.experiment}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Decision Function", "Kernel", "Slack", "Gamma", "Accuracy", "F-Measure", "Precision", "Recall", "Confusion Matrix"])

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
        y_val[y_val == 4] = 1
        y_test[y_test == 4] = 1


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
    elif args.experiment == 3:
        pass
    else:
        raise ValueError("Invalid experiment number")

    print(np.unique(y_train))
    extracted_X_train = feature_difference(X_train.copy())
    print(extracted_X_train.shape)
    extracted_X_val = feature_difference(X_val.copy())
    extracted_X_test = feature_difference(X_test.copy())

    del X_train
    del X_val
    del X_test

    # Param
    #Grid
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    dec_func = ['ovo', 'ovr']
    slack = [0.1, 1, 10, 100]
    gamma = ['auto', 'scale', 1, 0.1, 0.01, 0.001, 0.0001]

    weight = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    d_class_weights = dict(enumerate(weight))

    fm_ = -99999
    temp = 0
    for func in dec_func:
        for i in kernel:
            for c in slack:
                for g in gamma:
                    clf = SVC(decision_function_shape=func, kernel=i, C=c, gamma=g, class_weight=d_class_weights)
                    clf.fit(extracted_X_train, y_train)
                    prediction = clf.predict(extracted_X_test)

                    if args.experiment == 1 or args.experiment == 2:
                        acc = accuracy_score(y_test, prediction)
                        prec = precision_score(y_test, prediction)
                        rec = recall_score(y_test, prediction)
                        fm = f1_score(y_test, prediction)
                        fm_mean = fm
                        confus = confusion_matrix(y_test, prediction)
                    else:
                        acc = accuracy_score(y_test, prediction)
                        prec = precision_score(y_test, prediction, average='macro')
                        rec = recall_score(y_test, prediction, average='macro')
                        fm_mean = f1_score(y_test, prediction, average='macro')
                        confus = confusion_matrix(y_test, prediction)

                    print("Decision Function:", func)
                    print("Kernel:", i)
                    print("Slack:", c)
                    print("Gamma:", g)
                    print("F-Measure:", fm_mean)
                    print("Accuracy:", acc)
                    print("Precision:", prec)
                    print("Recall:", rec)
                    print(confus)

                    # Save results to CSV
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([func, i, c, g, acc, fm_mean, (prec), (rec), confus])

                    if fm_ < fm_mean:
                        best_func = func
                        best_kernel = i
                        best_slack = c
                        best_gamma = g
                        best_model = clf
                        fm_ = fm_mean
                        best_confus = confus

                    temp += 1
                    print(temp)

    print("Best Model:")
    print("Decision Function:", best_func)
    print("Kernel:", best_kernel)
    print("Slack:", best_slack)
    print("Gamma:", best_gamma)
    print("F-Measure:", fm_)

    # Define the directory path
    save_dir = "saved_model/Moya/"

    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    with open(f'saved_model/Moya/{fm_}_func_{best_func}_kernel_{best_kernel}_slack_{best_slack}_gamma_{best_gamma}_date_{datetime.now()}.pkl', 'wb') as outp:
        pickle.dump(best_model, outp, pickle.HIGHEST_PROTOCOL)

    # 2. Save detailed classification results
    results_dir = "results/Moya/"
    os.makedirs(results_dir, exist_ok=True)
    import pandas as pd
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

    # Save to CSV
    results_filename = f"{results_dir}Moya_best_model_results_exp_{args.experiment}.csv"
    results_dict.to_csv(results_filename, index=False)
    print(f"Saved classification results to: {results_filename}")