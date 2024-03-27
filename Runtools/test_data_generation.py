import sqlite3
import os
import random

# Make sure the working directory is correct
os.chdir("./Runtools/database")

# Connect to the database
conn = sqlite3.connect('database.db')
c = conn.cursor()

# Insert test data into `spsas` table
c.execute('''INSERT INTO spsas (spsa_alpha, spsa_gamma, spsa_c, spsa_A, spsa_a1) 
             VALUES (0.602, 0.101, 0.2, 10.0, 5.0)''')

# Insert test data into `feature_keys` table
c.execute('''INSERT INTO feature_keys (n_qubits, feature1, feature2) 
             VALUES (2, 'Z0', 'Z1')''')

# Insert test data into `data_sizes` table
c.execute('''INSERT INTO data_sizes (train_data_size, valid_data_size, test_data_size) 
             VALUES (100, 20, 30)''')

# Insert test data into `misc_params` table
c.execute('''INSERT INTO misc_params (n_epochs, batch_size, num_layers, obs, is_local_simulator, use_pca, seed) 
             VALUES (10, 5, 2, 'Z0Z1', 1, 0, 42)''')

# Assuming the `test_data` and `valid_loss` tables will be populated based on specific runs, we skip direct insertion for demonstration

# Insert test data into `outputs` table with foreign keys referencing the inserted test data
# Assuming IDs generated for `spsas`, `feature_keys`, `data_sizes`, and `misc_params` are 1 for simplicity
c.execute('''INSERT INTO outputs (spsas_id, feature_keys_id, data_sizes_id, misc_params_id, cost, test_accuracy, run_time) 
             VALUES (1, 1, 1, 1, 0.5, 0.95, 60.0)''')


# Insert 4 circuits into `outputs` table (this assumes previous insertions have been made)
for _ in range(4):
    c.execute('''INSERT INTO outputs (spsas_id, feature_keys_id, data_sizes_id, misc_params_id, cost, test_accuracy, run_time) 
                 VALUES (1, 1, 1, 1, 0.5, 0.95, 60.0)''')

# Insert 10 random data points into `valid_loss` table for each circuit
for circuit_id in range(1, 5):
    for _ in range(10):
        valid_loss_value = random.random()  # Generating a random float between 0 and 1
        c.execute('''INSERT INTO valid_loss (circuit_id, valid_loss) 
                     VALUES (?, ?)''', (circuit_id, valid_loss_value))


# Commit the changes to the database
conn.commit()

# Close the connection
conn.close()

print("Test data inserted into the database successfully.")
