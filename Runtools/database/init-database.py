import sqlite3

conn = sqlite3.connect('database.db')

c = conn.cursor()

c.execute('''CREATE TABLE spsas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spsa_alpha FLOAT,
    spsa_gamma FLOAT,
    spsa_c FLOAT,
    spsa_A FLOAT,
    spsa_a1 FLOAT
)''')

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

c.execute('''CREATE TABLE data_sizes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    train_data_size INTEGER,
    valid_data_size INTEGER,
    test_data_size INTEGER
)''')

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

c.execute('''CREATE TABLE test_data (
    id INTEGER PRIMARY KEY,
    test_label INTEGER,
    test_prob FLOAT,
    test_pred INTEGER
)''')

c.execute('''CREATE TABLE valid_loss (
    id INTEGER PRIMARY KEY,
    valid_loss FLOAT
)''')

c.execute('''CREATE TABLE outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FOREIGN KEY (spsa_id) REFERENCES spsas(id),
    FOREIGN KEY (feature_key_id) REFERENCES feature_keys(id),
    FOREIGN KEY (data_size_id) REFERENCES data_sizes(id),
    FOREIGN KEY (misc_param_id) REFERENCES misc_params(id),
    cost FLOAT,
    test_accuracy FLOAT,
    FOREIGN KEY (test_data_id) REFERENCES test_data(id),
    FOREIGN KEY (valid_loss_id) REFERENCES valid_loss(id),
    run_time FLOAT
)''')