# IMPORTS
import numpy as np
from qiskit_aer.primitives import Sampler, Estimator
# from qiskit.primitives import Estimator



import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import os

import Modules.LHC_QML_module as lqm




def run(params, data, qc_template):
    
    """
    We import and rename all parameters here to allow for easy changes to
    this model without having to reference the new parameters.


    """

    # SETTINGS TO TUNE (IMPORTED FROM ELSEWHERE)
    seed = params.seed
    save_folder = params.save_folder
    batch_size = params.batch_size
    n_epochs = params.n_epochs

    train_data_size = params.train_data_size  
    test_data_size = params.test_data_size  
    valid_data_size = params.valid_data_size  
    total_datasize = train_data_size + test_data_size + valid_data_size

    # Assuming optimizer settings are also to be parameterized
    # opt = NesterovMomentumOptimizer(params.optimizer_learning_rate) 
    # opt = SPSAOptimizer(maxiter=params.optimizer_maxiter) 

    is_local_simulator = params.is_local_simulator 
    n_qubits = params.n_qubits  
    num_layers = params.num_layers  

    spsa_alpha = params.spsa_alpha
    spsa_gamma = params.spsa_gamma
    spsa_c = params.spsa_c
    spsa_A = params.spsa_A
    spsa_a = params.spsa_a

    par_inputs = params.par_inputs
    par_weights = params.par_weights
    obs = params.obs


    # IMPORT TRAINING/TESTING DATA FROM ELSEWHERE
    train_features = data.train_features
    train_labels = data.train_labels

    valid_features = data.valid_features
    valid_labels =  data.valid_labels

    test_features = data.test_features
    test_labels = data.test_labels




    np.random.seed(seed)

    """
    if os.path.exists(save_folder):
        print(f"This notebook may overwrite previous results in the {save_folder} directory")
    else:
        os.makedirs(save_folder)
    """

    # weights_init = 0.5 * np.random.randn(num_layers, n_qubits, requires_grad=True)
    weights_init = 0.5 * np.random.randn(num_layers , n_qubits)
    weights_init = weights_init.flatten()



    # DEFINE LOSS
    def loss(prob, label):
        # print(prob)
        return -np.mean(label*np.log(prob+1e-5)+(1-label)*np.log(1-prob+1e-5))

    def accuracy(pred, label):
        return np.mean(np.isclose(pred,label))

    # def cost(weights, features, labels):
    #     probs = np.array([model(weights, f) for f in features])
    #     return loss(probs, labels)


    


    # TRAINING

    num_train=train_features.shape[0]
    weights = weights_init
    n_batches = num_train // batch_size


    # losses = []
    times = []
    losses_valid = []

    start = time.time()
    times.append(start)

    spsa_k = 0

    estimator = Estimator()

    for i in range(n_epochs):
        indices = list(range(num_train))
        np.random.shuffle(indices)
        
        for j in range(n_batches):
            # Update the weights by one optimizer step
            batch_index = indices[j*batch_size:(j+1)*batch_size]
            # batch_index = np.random.randint(0, num_train, (batch_size,))
            train_features_batch = train_features[batch_index]
            train_labels_batch = train_labels[batch_index]

            spsa_k += 1

            spsa_ck = spsa_c / spsa_k ** spsa_gamma
            weights_deltas = np.random.choice([-1, 1], size = weights.shape) * spsa_ck
            weights_plus = weights + weights_deltas
            weights_minus = weights - weights_deltas

            qc_plus_list = [qc_template.assign_parameters({par_weights: weights_plus, par_inputs: input}) for input in train_features_batch]
            qc_minus_list = [qc_template.assign_parameters({par_weights: weights_minus, par_inputs: input}) for input in train_features_batch]
            

            qc_list = qc_plus_list + qc_minus_list
            obs_list = [obs] * (2*batch_size)
            
            job = estimator.run(qc_list, obs_list, shots=1024)
            result = job.result()
            # the results are between -1 and 1
            # rescale it to 0 to 1
            probs_all = (np.array(result.values) + 1) / 2

            probs_plus = probs_all[:batch_size]
            probs_minus = probs_all[-batch_size:]

            loss_plus = loss(probs_plus, train_labels_batch)
            loss_minus = loss(probs_minus, train_labels_batch)

            grad = (loss_plus - loss_minus) /2 / weights_deltas

            spsa_ak = spsa_a / (spsa_A + spsa_k) ** spsa_alpha

            weights -= spsa_ak * grad

            ##np.savez(os.path.join(save_folder, f"weights_{i}_{j}"), weights=weights) TEMP DO NOT SAVE

            times.append(time.time())
            delta_t = times[-1]-times[-2]

            if is_local_simulator: # would be too costly on a real qpu
                qc_valid_list = [qc_template.assign_parameters({par_weights:weights, par_inputs: f}) for f in valid_features]
                obs_list = [obs] * len(qc_valid_list)
                
                job = estimator.run(qc_valid_list, obs_list, shots=1024)
                result = job.result()
                
                probs_valid = (np.array(result.values) + 1 ) / 2
                predictions_val = np.round(probs_valid)
                acc_valid = accuracy(valid_labels, predictions_val)
                cost_valid = loss(probs_valid, valid_labels)

                losses_valid.append(cost_valid)
        
            message = f"Epoch: {i+1:4d} | Iter: {j+1:4d}/{n_batches} | Time: {delta_t:0.2f} |" 
            if is_local_simulator:
                message += f" Cost val: {cost_valid:0.3f} | Acc val:  {acc_valid:0.3f}"
            print(message)







    # TESTING
    start = time.time()

    qc_test_list = [qc_template.assign_parameters({par_weights:weights, par_inputs: f}) for f in test_features]
    obs_list = [obs] * len(qc_test_list)

    job = estimator.run(qc_test_list, obs_list, shots=1024)
    result = job.result()
    probs_test = (np.array(result.values) + 1 ) / 2
    preds_test = np.round(probs_test)

    message = ""

    elapsed = time.time() - start
    message += f"Testing time: {round(elapsed)} seconds\n"

    cost_test = loss(probs_test, test_labels)
    acc_test = accuracy(preds_test, test_labels)

    message += f"Test cost: {cost_test:0.3f} | Test accuracy: {acc_test:0.3f} |"






    # GRAPHS AND CONFUSION MATRIX
    # Plot the loss TEMP DISABLED
    if is_local_simulator:
        lqm.plot_loss(losses_valid)
        plt.savefig("validation_loss.png")

    # Plot the class histogram (probability of being classified as signal)
    lqm.plot_class_hist(probs_test, test_labels)
    plt.ylim([0, 16])
    plt.title("Ideal simulation")
    plt.savefig("classhist.png")

    # Plot the ROC curve
    lqm.plot_roc(probs_test, test_labels)
    plt.title("Ideal simulation")
    plt.savefig("roc.png")

    # Confusion matrix
    cm = confusion_matrix(test_labels, preds_test)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("confusion_matrix.png")

    message += "\n\n\n\n\n"

    # also print a table in the markdown format
    message += "| | predict 0 | predict 1|\n"
    message += "|---|---|---|\n"
    message += f"|true 0|  {cm[0, 0]} | {cm[0, 1]} |\n"
    message += f"|true 1|  {cm[1, 0]} | {cm[1, 1]} |\n"






    # PRINT RESULTS
    message += "\n\n\n\n\n"


    message += "| range | n signal | n background | percentage signal  | percentage background| \n"
    message += "|---|---|---|---|---|\n"

    cuts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


    for cut in cuts:
        filter = (probs_test > cut)
        filtered_labels = test_labels[filter]

        n_total = len(filtered_labels)
        n_signal = np.sum(filtered_labels).astype(int)
        n_background = n_total-n_signal
        
        if n_total != 0:
            per_signal = n_signal/n_total * 100
            per_background = n_background/n_total * 100

        message += f"| >{cut:.1f} | {n_signal} | {n_background} | {per_signal:.1f}%  |  {per_background:.1f}% | \n"

    print(message)
    with open("results.txt", "w") as f:
        f.write(message)