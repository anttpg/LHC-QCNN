# IMPORTS
import numpy as np
from qiskit_aer.primitives import Sampler, Estimator
# from qiskit.primitives import Estimator



import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import Modules.LHC_QML_module as lqm




def run(params, data, qc_template):
    
    np.random.seed(params.seed)

    """
    if os.path.exists(save_folder):
        print(f"This notebook may overwrite previous results in the {save_folder} directory")
    else:
        os.makedirs(save_folder)
    """

    # weights_init = 0.5 * np.random.randn(num_layers, n_qubits, requires_grad=True)
    weights_init = 0.5 * np.random.randn(params.num_layers , params.n_qubits)
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

    num_train=data.train_features.shape[0]
    weights = weights_init
    n_batches = num_train // params.batch_size


    # losses = []
    times = []
    losses_valid = []

    start = time.time()
    times.append(start)

    spsa_k = 0

    estimator = Estimator()

    for i in range(params.n_epochs):
        indices = list(range(num_train))
        np.random.shuffle(indices)
        
        for j in range(n_batches):
            # Update the weights by one optimizer step
            batch_index = indices[j*params.batch_size:(j+1)*params.batch_size]
            # batch_index = np.random.randint(0, num_train, (batch_size,))
            train_features_batch = data.train_features[batch_index]
            train_labels_batch = data.train_labels[batch_index]

            spsa_k += 1

            spsa_ck = params.spsa_c / spsa_k ** params.spsa_gamma
            weights_deltas = np.random.choice([-1, 1], size = weights.shape) * spsa_ck
            weights_plus = weights + weights_deltas
            weights_minus = weights - weights_deltas

            qc_plus_list = [qc_template.assign_parameters({params.par_weights: weights_plus, params.par_inputs: input}) for input in train_features_batch]
            qc_minus_list = [qc_template.assign_parameters({params.par_weights: weights_minus, params.par_inputs: input}) for input in train_features_batch]
            

            qc_list = qc_plus_list + qc_minus_list
            obs_list = [params.obs] * (2*params.batch_size)
            
            job = estimator.run(qc_list, obs_list, shots=1024)
            result = job.result()
            # the results are between -1 and 1
            # rescale it to 0 to 1
            probs_all = (np.array(result.values) + 1) / 2

            probs_plus = probs_all[:params.batch_size]
            probs_minus = probs_all[-params.batch_size:]

            loss_plus = loss(probs_plus, train_labels_batch)
            loss_minus = loss(probs_minus, train_labels_batch)

            grad = (loss_plus - loss_minus) /2 / weights_deltas

            spsa_ak = params.spsa_a / (params.spsa_A + spsa_k) ** params.spsa_alpha

            weights -= spsa_ak * grad

            ##np.savez(os.path.join(save_folder, f"weights_{i}_{j}"), weights=weights) TEMP DO NOT SAVE

            times.append(time.time())
            delta_t = times[-1]-times[-2]

            if params.is_local_simulator: # would be too costly on a real qpu
                qc_valid_list = [qc_template.assign_parameters({params.par_weights:weights, params.par_inputs: f}) for f in data.valid_features]
                obs_list = [params.obs] * len(qc_valid_list)
                
                job = estimator.run(qc_valid_list, obs_list, shots=1024)
                result = job.result()
                
                probs_valid = (np.array(result.values) + 1 ) / 2
                predictions_val = np.round(probs_valid)
                acc_valid = accuracy(data.valid_labels, predictions_val)
                cost_valid = loss(probs_valid, data.valid_labels)

                losses_valid.append(cost_valid)
        
            # TODO: Change this eventually to a logging system instead of print
            message = f"Epoch: {i+1:4d} | Iter: {j+1:4d}/{n_batches} | Time: {delta_t:0.2f} |" 
            if params.is_local_simulator:
                message += f" Cost val: {cost_valid:0.3f} | Acc val:  {acc_valid:0.3f}"
            print(message)







    # TESTING
    start = time.time()

    qc_test_list = [qc_template.assign_parameters({params.par_weights:weights, params.par_inputs: f}) for f in data.test_features]
    obs_list = [params.obs] * len(qc_test_list)

    job = estimator.run(qc_test_list, obs_list, shots=1024)
    result = job.result()
    probs_test = (np.array(result.values) + 1 ) / 2
    preds_test = np.round(probs_test)

    message = ""

    elapsed = time.time() - start
    message += f"Testing time: {round(elapsed)} seconds\n"

    cost_test = loss(probs_test, data.test_labels)
    acc_test = accuracy(preds_test, data.test_labels)

    message += f"Test cost: {cost_test:0.3f} | Test accuracy: {acc_test:0.3f} |"






    # GRAPHS AND CONFUSION MATRIX
    # Plot the loss TEMP DISABLED
    if params.is_local_simulator:
        lqm.plot_loss(losses_valid)
        plt.savefig("validation_loss.png")

    # Plot the class histogram (probability of being classified as signal)
    lqm.plot_class_hist(probs_test, data.test_labels)
    plt.ylim([0, 16])
    plt.title("Ideal simulation")
    plt.savefig("classhist.png")

    # Plot the ROC curve
    lqm.plot_roc(probs_test, data.test_labels)
    plt.title("Ideal simulation")
    plt.savefig("roc.png")

    # Confusion matrix
    cm = confusion_matrix(data.test_labels, preds_test)
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
        filtered_labels = data.test_labels[filter]

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