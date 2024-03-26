import sqlite3
import os

# os.chdir("./Runtools/database")
conn = sqlite3.connect('database.db')

c = conn.cursor()
# DATABASE.DB IS IN GITIGNORE, MUST RUN THIS TO CREATE LOCALLY

# Table for spsa parameters. Each set of parameters should have a unique id.
# Before adding a new set of parameters, check if the parameters already exist in the table.
# If the parameters already exist, use the id of the existing parameters as the foreign key for your new outputs table entry.
c.execute('''CREATE TABLE spsas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spsa_alpha FLOAT,
    spsa_gamma FLOAT,
    spsa_c FLOAT,
    spsa_A FLOAT,
    spsa_a1 FLOAT
)''')

# Table for feature keys and number of qubits. Combined because they are related. Each set of keys should have a unique id.
# Before adding a new set of keys, check if the keys already exist in the table.
# If the keys already exist, use the id of the existing keys as the foreign key for your new outputs table entry.
c.execute('''CREATE TABLE feature_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    n_qubits INTEGER,
    feature1 TEXT NULL,
    feature2 TEXT NULL,
    feature3 TEXT NULL,
    feature4 TEXT NULL,
    feature5 TEXT NULL,
    feature6 TEXT NULL,
    feature7 TEXT NULL,
    feature8 TEXT NULL,
    feature9 TEXT NULL,
    feature10 TEXT NULL,
    feature11 TEXT NULL,
    feature12 TEXT NULL,
    feature13 TEXT NULL,
    feature14 TEXT NULL,
    feature15 TEXT NULL,
    feature16 TEXT NULL,
    feature17 TEXT NULL,
    feature18 TEXT NULL,
    feature19 TEXT NULL,
    feature20 TEXT NULL,
    feature21 TEXT NULL,
    feature22 TEXT NULL,
    feature23 TEXT NULL,
    feature24 TEXT NULL,
    feature25 TEXT NULL
)''')

# Table for data size parameters. Each set of parameters should have a unique id.
# Before adding a new set of parameters, check if the parameters already exist in the table.
# If the parameters already exist, use the id of the existing parameters as the foreign key for your new outputs table entry.
c.execute('''CREATE TABLE data_sizes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    train_data_size INTEGER,
    valid_data_size INTEGER,
    test_data_size INTEGER
)''')

# Table for miscellaneous parameters. Each set of parameters should have a unique id.
# Before adding a new set of parameters, check if the parameters already exist in the table.
# If the parameters already exist, use the id of the existing parameters as the foreign key for your new outputs table entry.
c.execute('''CREATE TABLE misc_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    n_epochs INTEGER,
    batch_size INTEGER,
    num_layers INTEGER,
    obs TEXT,
    is_local_simulator BOOLEAN,
    use_pca BOOLEAN,
    seed INTEGER
)''')

# Table for test data. Lists of test labels, probabilities, and predictions that are x entries long
# should be represented as x rows in this table (they will all be the same size for some run). 
# All of those rows should have the same circuit_id (id must be unique). The id of the test data should be used as the foreign key 
# for your new outputs table entry. The id will be how we keep track of which test data belongs to which outputs.
c.execute('''CREATE TABLE test_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit_id INTEGER,
    test_label INTEGER,
    test_prob FLOAT,
    test_pred INTEGER
)''')

# Table for validation loss. A list of losses that is x entries long
# should be represented as x rows in this table. All of those rows should have the same circuit_id (id must be unique).
# The id of the losses should be used as the foreign key for your new outputs table entry.
# The id will be how we keep track of which loss data belongs to which outputs.
c.execute('''CREATE TABLE valid_loss (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit_id INTEGER,
    valid_loss FLOAT
)''')


# THE LENGTH OF THIS WILL NEED TO BE CHANGED BASED ON THE CIRCUIT THAT IT IS BEING USED FOR
# epoch == 0 will be the initial weights
c.execute('''CREATE TABLE parameter_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit_id INTEGER,
    epoch INTEGER,
    batch INTEGER,
    w1 FLOAT,
    w2 FLOAT,
    w3 FLOAT,
    w4 FLOAT,
    w5 FLOAT,
    w6 FLOAT,
    w7 FLOAT,
    w8 FLOAT,
    w9 FLOAT,
    w10 FLOAT,
    w11 FLOAT,
    w12 FLOAT,
    w13 FLOAT,
    w14 FLOAT,
    w15 FLOAT
)''')



# This will be the main database table. It will contain all of the outputs from the runs.
# It will also contain references to the other tables so that we can keep track of which outputs belong to which parameters.
# When querying for data to analyze, we will be able to join the tables on the foreign keys to get all of the data we need.
c.execute('''CREATE TABLE outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spsas_id INTEGER,
    feature_keys_id INTEGER,
    data_sizes_id INTEGER,
    misc_params_id INTEGER,
    test_data_id INTEGER,
    valid_loss_id INTEGER,
    cost FLOAT,
    test_accuracy FLOAT,
    run_time FLOAT,
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (spsas_id) REFERENCES spsas(id),
    FOREIGN KEY (feature_keys_id) REFERENCES feature_keys(id),
    FOREIGN KEY (data_sizes_id) REFERENCES data_sizes(id),
    FOREIGN KEY (misc_params_id) REFERENCES misc_params(id)
)''')
    # No point in doing this, just use the id of the outputs table to query the other tables (joining creates a new data row for every test data row)
    # FOREIGN KEY (test_data_id) REFERENCES test_data(circuit_id),
    # FOREIGN KEY (valid_loss_id) REFERENCES valid_loss(circuit_id)