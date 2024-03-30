import hashlib
import json
import sys
import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
sys.path.append("./../")

from database import Database
import Modules.LHC_QML_module as lqm

DATABASE_PATH1 = "./../database/database.db"
DATABASE_PATH2 = "./../database/circuit1.db"
db = Database(DATABASE_PATH1, None, None, False)
db1 = Database(DATABASE_PATH2, None, None, False)
# Use old feature keys
# Implement noise model
# Use new circuit
# Eventually try implementing trainable coefficient for parameters
# Write code to show discrimination for best hyperparameter combinations
# Find moments (mean, variance, skew) of test accuracy histograms
# Test same initial parameters and hyperparameters to see if they converge to the same values
LAST_RUNS = db.get_conditional_data(feature_keys=['f_lept3_pt', 'f_lept4_pt', 'f_Z1mass'])
C1 = db1.get_conditional_data(feature_keys=['f_lept3_pt', 'f_lept4_pt', 'f_Z1mass'])


def generate_hash(parameters):
    # Convert parameters dictionary to a string
    parameters_str = json.dumps(parameters, sort_keys=True)
    
    # Compute the SHA-256 hash
    hash_object = hashlib.sha256(parameters_str.encode())
    hash_str = hash_object.hexdigest()
    
    return hash_str


def get_spsas_given_fks(fks, ds, D):
    if fks:
        data = D.get_conditional_data(feature_keys=fks)
    else:
        # data = db.get_conditional_data(data_sizes=ds)
        data = LAST_RUNS
    plot_data = {}
    spsas = {}
    alpha = {}
    gamma = {}
    c = {}
    A = {}
    a1 = {}
    for d in data:
        if d["test_accuracy"]:
            h = generate_hash(d["spsas"])
            plot_data[h] = d["test_accuracy"]
            spsas[h] = d["spsas"]
            alpha[d["spsas"]["spsa_alpha"]] = d["test_accuracy"]
            gamma[d["spsas"]["spsa_gamma"]] = d["test_accuracy"]
            c[d["spsas"]["spsa_c"]] = d["test_accuracy"]
            A[d["spsas"]["spsa_A"]] = d["test_accuracy"]
            a1[d["spsas"]["spsa_a1"]] = d["test_accuracy"]
    spsass = list("alpha: " + str(spsas[k]["spsa_alpha"]) + "\n gamma: " + str(spsas[k]["spsa_gamma"]) + "\n c: " + str(spsas[k]["spsa_c"]) + "\n A: " + str(spsas[k]["spsa_A"]) + "\n a1: " + str(spsas[k]["spsa_a1"]) for k in plot_data.keys())
    return plot_data, spsas, alpha, gamma, c, A, a1, spsass


def plot_hist_above_accuracy(data, test_acc, spsa_string, bins=15):
    goodkeys = list(k for k in data.keys() if data[k] > test_acc)
    plt.hist(goodkeys, bins=bins)
    # plt.title(spsa_string + " vs Test accuracy")
    # plt.ylabel("test accuracy")
    # plt.xlabel(spsa_string)
    # plt.show()


def plot_data_vs_acc(data, strings):
    data = dict(sorted(data.items()))
    plt.plot(data.keys(), data.values())
    plt.xticks(list(data.keys()), strings)
    # plt.xticks([], strings)


def normalize_value(value, min_value, max_value, new_min, new_max):
    return list(((v - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min for v in value)



alpha_ranges = [(0.1, 0.4), (0.4, 0.7), (0.7, 1)]
# alpha_ranges = [(0.485, 0.5), (0.5, 0.515)]
gamma_ranges = [(0.071, 0.091), (0.091, 0.111), (0.111, 0.131)]
# gamma_ranges = [(0.091, 0.101), (0.101, 0.111)]
c_a1_ranges = [(0.05, 0.2), (0.2, 0.35), (0.35, 0.5)]
# c_a1_ranges = [(0.190, 0.2025), (0.2025, 0.215)]
A_ranges = [(1.5, 1.83), (1.83, 2.16), (2.16, 2.5)]
# A_ranges = [(1.975, 2), (2, 2.025)]
def get_best_spsa_ranges(data, spsas, test_acc):

    spsa_range_amts = {}
    for key in data:
        if data[key] > test_acc:
            ranges_list = [0, 0, 0, 0, 0]
            for i in range(len(alpha_ranges)):
                if alpha_ranges[i][0] <= spsas[key]["spsa_alpha"] <= alpha_ranges[i][1]:
                    ranges_list[0] = i
                if gamma_ranges[i][0] <= spsas[key]["spsa_gamma"] <= gamma_ranges[i][1]:
                    ranges_list[1] = i
                if c_a1_ranges[i][0] <= spsas[key]["spsa_c"] <= c_a1_ranges[i][1]:
                    ranges_list[2] = i
                if c_a1_ranges[i][0] <= spsas[key]["spsa_a1"] <= c_a1_ranges[i][1]:
                    ranges_list[4] = i
                if A_ranges[i][0] <= spsas[key]["spsa_A"] <= A_ranges[i][1]:
                    ranges_list[3] = i
            ranges_tup = tuple(ranges_list)
            if ranges_tup in spsa_range_amts:
                spsa_range_amts[ranges_tup] += 1
            else:
                spsa_range_amts[ranges_tup] = 1
    return spsa_range_amts, max(spsa_range_amts.values())




def plot_average_loss(data):

    total_loss_1ep = np.zeros(200)
    total_loss_2ep = np.zeros(400)
    num_1ep = 0
    for d in data:
        loss = np.array(d["valid_loss"])
        if d["misc_params"]["n_epochs"] == 1:
            total_loss_1ep += loss
            num_1ep += 1
        else:
            total_loss_2ep += loss

    loss_1ep, loss_2ep = total_loss_1ep / num_1ep, total_loss_2ep / (len(data) - num_1ep)
    
    plt.plot(loss_1ep)
    plt.title("Average Validation Loss for 1 Epoch Runs")
    plt.show()
    plt.plot(loss_2ep)
    plt.title("Average Validation Loss for 2 Epoch Runs")
    plt.show()


def test_acc_hists(data, title, bins=15):
    test_accs = []
    for d in data:
        if d["test_accuracy"]:
            test_accs.append(d["test_accuracy"])
    plt.hist(test_accs, bins=bins)
    plt.title("Test accuracy " + title)
    plt.show()


def class_hist(data, title):
    labels = []
    probs = []
    for d in data:
        if d["test_accuracy"]:
            labels += d["test_labels"]
            probs += d["test_probs"]
    
    lqm.plot_class_hist(np.array(probs), np.array(labels))
    plt.title("Class histogram (probability of being classified as signal) " + title)
    plt.show()


def plot_confusion_matrix(data):
    labels = []
    preds = []

    for d in data:
        if d["test_accuracy"]:
            labels += d["test_labels"]
            preds += d["test_preds"]
    
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()


def plot_roc(data, title):
    labels = []
    probs = []
    for d in data:
        if d["test_accuracy"]:
            labels += d["test_labels"]
            probs += d["test_probs"]
    
    lqm.plot_roc(np.array(probs), np.array(labels))
    plt.title("ROC curve " + title)
    plt.show()



def plot_for_feature_keys(fks):
    db.plot_datapoints(fks)
    data = db.get_conditional_data(feature_keys=fks, data_sizes=(400, 50, 150))

    test_acc_hists(data)
    plot_average_loss(data)
    class_hist(data)
    plot_roc(data)
    confusion_matrix(data)



test_acc_hists(LAST_RUNS, "New Circuit (less layers)")
test_acc_hists(C1, "Old Circuit (more layers)")

# plot_average_loss(LAST_RUNS)

class_hist(LAST_RUNS, "New Circuit (less layers)")
class_hist(C1, "Old Circuit (more layers)")

plot_roc(LAST_RUNS, "New Circuit (less layers)")
plot_roc(C1, "Old Circuit (more layers)")

plot_confusion_matrix(LAST_RUNS)
plot_confusion_matrix(C1)




# Show best parameter groupings
for el in [db, db1]:

    plot_data, spsas, alpha, gamma, c, A, a1, spsass = get_spsas_given_fks(['f_lept3_pt', 'f_lept4_pt', 'f_Z1mass'], None, el)

    if el == db:
        print("New Circuit (less layers)")
    else:
        print("Old Circuit (more layers)")

    ranges, max_ = get_best_spsa_ranges(plot_data, spsas, 0.8)

    for k in ranges:
        amt_total = 0
        if ranges[k] == max_ or ranges[k] == max_ - 1:
            for data in LAST_RUNS:
                if data["test_accuracy"]:
                    if alpha_ranges[k[0]][0] <= data["spsas"]["spsa_alpha"] <= alpha_ranges[k[0]][1] and gamma_ranges[k[1]][0] <= data["spsas"]["spsa_gamma"] <= gamma_ranges[k[1]][1] and c_a1_ranges[k[2]][0] <= data["spsas"]["spsa_c"] <= c_a1_ranges[k[2]][1] and c_a1_ranges[k[4]][0] <= data["spsas"]["spsa_a1"] <= c_a1_ranges[k[4]][1] and A_ranges[k[3]][0] <= data["spsas"]["spsa_A"] <= A_ranges[k[3]][1]:
                        amt_total += 1
            print(("alpha: " + str(alpha_ranges[k[0]]), "gamma: " + str(gamma_ranges[k[1]]), "c: " + str(c_a1_ranges[k[2]]), "A: " + str(A_ranges[k[3]]), "a1: " + str(c_a1_ranges[k[4]]), "Total > 0.8: " + str(ranges[k]), "Total: " + str(amt_total)))

    removals = []
    for key in ranges:
        if ranges[key] < 2:
            removals.append(key)
    for r in removals:
        ranges.pop(r)

    plt.bar(range(len(ranges)), list(ranges.values()), align='center')
    plt.xticks(range(len(ranges)), list(ranges.keys()))
    plt.title("Parameter groupings with test accuracy > 0.8")
    plt.show()