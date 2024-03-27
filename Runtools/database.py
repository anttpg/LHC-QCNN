from sklearn.model_selection import train_test_split
import Modules.LHC_QML_module as lqm
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import sqlite3
from functools import partial
import traceback
import queue
import time

class Database:

    def __init__(self, filepath, request_queue, output_queue, use_interface):
        self.request_queue = request_queue
        self.output_queue = output_queue
    
        try:
            self.conn = sqlite3.connect(filepath)
        except Exception as e:
            traceback.print_exc()
            print("\n\n")
            print(os.listdir(filepath))
            print("Error loading database from \'" + filepath + "\', likely db is not generated. DATABASE.DB IS IN GITIGNORE, MUST RUN 'init-database.py' TO CREATE LOCALLY")
            exit(0)

        self.cursor = self.conn.cursor()



        if use_interface:
            print("! Call process requests !")
            self.process_requests()




    # Close and save database
    def close(self):
        self.conn.close()



    def plot_datapoints(self):
        signal_dict, background_dict, files_used = lqm.load_data(
            self.params.signals_folder, self.params.backgrounds_folder, self.params.training_feature_keys
        )

        lqm.plot_pairwise_dicts(signal_dict, background_dict)
        plt.savefig(self.id + "dataplot.png")
        plt.close("all")



    # FUNS TO IMPLEMENT, I LEAVE THE SPECIFICS TO YOU.   
    # SOMEHOW MAKE IT SO Only ONE INSTANCE OF THIS CLASS CAN BE MADE. (__new__ function or something of the like? Search online.)

    # MAKE A DICTIONARY of (circuit_id -> [plot, func] list) that will be checked every time the callback called.
    plots = []






    def select_spsas(self, spsa_alpha=None, spsa_gamma=None, spsa_c=None, spsa_A=None, spsa_a1=None):
        """
        This function selects the id of the given spsa parameters from the database. Or None if the parameters do not exist. 

        Args:
            spsa_alpha (float): The alpha parameter for spsa.
            spsa_gamma (float): The gamma parameter for spsa.
            spsa_c (float): The c parameter for spsa.
            spsa_A (float): The A parameter for spsa.
            spsa_a1 (float): The a1 parameter for spsa.
        
        Returns:
            list: The id of the spsa parameters if they exist (possibly multiple if <5 parameters were entered), otherwise empty list.
            Throws an exception if no parameters are given.
        """
        # Allow parameters to be optional so that we can check different combinations of parameters
        # If a parameter is not None, add it to the message and the parameters list
        # IMPORTANT: Cannot have all parameters be None
        message = "SELECT id FROM spsas WHERE "
        params = []

        # Add a ? for each parameter that is not None
        if spsa_alpha is not None:
            message += "spsa_alpha = ? AND "
            params.append(spsa_alpha)
        if spsa_gamma is not None:
            message += "spsa_gamma = ? AND "
            params.append(spsa_gamma)
        if spsa_c is not None:
            message += "spsa_c = ? AND "
            params.append(spsa_c)
        if spsa_A is not None:
            message += "spsa_A = ? AND "
            params.append(spsa_A)
        if spsa_a1 is not None:
            message += "spsa_a1 = ? AND "
            params.append(spsa_a1)

        # Remove the last " AND " from the message
        message = message[:-5]

        # If no parameters were given, raise an exception
        if len(params) == 0:
            Exception("No parameters given to select_spsas")

        self.cursor.execute(message, (*params,))
        # Will be none if the parameters do not exist, otherwise will be tuples of the ids of the parameters inside a list
        return self.cursor.fetchall()







    def select_feature_keys(self, n_qubits, feature_keys):
        """
        This function selects the id of the given feature keys from the database. Or None if the keys do not exist. 
        IMPORTANT: feature_keys must be in the same order as they are in the database (ie same as the order of the list the circuit was run with)
        Easiest way to keep track of this may be to always ensure the list is sorted alphabetically before being passed to a circuit or this function.

        Args:
            n_qubits (int): The number of qubits the feature keys are for.
            feature_keys (list): The feature keys to select the id of.

        Returns:
            int: The id of the feature keys if they exist, otherwise None.
        """
        message = "SELECT id FROM feature_keys WHERE n_qubits = ? AND "
        # Add a ? for each feature key
        for i in range(1, len(feature_keys) + 1):
            message += f"feature{i} = ? AND "
        # Remove the last " AND " from the message
        message = message[:-5]

        # Execute the message with the n_qubits and feature keys as parameters
        self.cursor.execute(message, (n_qubits, *feature_keys))

        # Will be none if the parameters do not exist, otherwise will be the id of the parameters inside a list
        return self.cursor.fetchall()






    def select_data_sizes(self, train_data_size, valid_data_size, test_data_size):
        """
        This function selects the id of the given data sizes from the database. Or None if the sizes do not exist. 

        Args:
            train_data_size (int): The size of the training data.
            valid_data_size (int): The size of the validation data.
            test_data_size (int): The size of the test data.

        Returns:
            int: The id of the data sizes if they exist in the database, otherwise None.
        """
        self.cursor.execute("SELECT id FROM data_sizes WHERE train_data_size = ? AND valid_data_size = ? AND test_data_size = ?", 
                            (train_data_size, valid_data_size, test_data_size))
        # Will be none if the parameters do not exist, otherwise will be the id of the parameters inside a list
        return self.cursor.fetchall()






    def select_misc_params(self, n_epochs=None, batch_size=None, num_layers=None, obs=None, is_local_simulator=None, use_pca=None, seed=None):
        """
        This function selects the id of the given misc parameters from the database. Or None if the parameters do not exist. 

        Args:
            n_epochs (int): The number of epochs.
            batch_size (int): The batch size.
            num_layers (int): The number of layers.
            obs (str): The obs string.
            is_local_simulator (bool): Whether the simulator is local.
            use_pca (bool): Whether to use pca.
            seed (int): The seed.

        Returns:
            int: The list (possibly multiple if <5 params entered) of ids of the misc parameters if they exist in the database, otherwise empty list.
            Throws an exception if no parameters are given.
        """
        message = "SELECT id FROM misc_params WHERE "
        params = []

        # Add a ? for each parameter that is not None
        if n_epochs is not None:
            message += "n_epochs = ? AND "
            params.append(n_epochs)
        if batch_size is not None:
            message += "batch_size = ? AND "
            params.append(batch_size)
        if num_layers is not None:
            message += "num_layers = ? AND "
            params.append(num_layers)
        if obs is not None:
            message += "obs = ? AND "
            params.append(obs)
        if is_local_simulator is not None:
            message += "is_local_simulator = ? AND "
            params.append(is_local_simulator)
        if use_pca is not None:
            message += "use_pca = ? AND "
            params.append(use_pca)
        if seed is not None:
            message += "seed = ? AND "
            params.append(seed)

        # Remove the last " AND " from the message
        message = message[:-5]

        # If no parameters were given, raise an exception
        if len(params) == 0:
            Exception("No parameters given to select_misc_params")

        self.cursor.execute(message, (*params,))

        # Will be none if the parameters do not exist, otherwise will be a list of tuples of the ids of the parameters
        return self.cursor.fetchall()







    def create_circuit_entry(self, params):
        """
        Create a new entry in the circuits table with the given parameters. Return the id of the new entry as the unique id for the current circuit.

        Args:
            params (Parameters): The parameters for the circuit.

        Returns:
            int: The id of the new entry in the circuits table.
        """

        # Check if the spsa params already exist in the table
        spsas_id = self.select_spsas(params.spsa_alpha, params.spsa_gamma, params.spsa_c, params.spsa_A, params.spsa_a1)
        # If the spsa params do not exist in the table, add them and get the new id
        if not spsas_id:
            self.cursor.execute("INSERT INTO spsas (spsa_alpha, spsa_gamma, spsa_c, spsa_A, spsa_a1) VALUES (?, ?, ?, ?, ?)", 
                                (params.spsa_alpha, params.spsa_gamma, params.spsa_c, params.spsa_A, params.spsa_a1))
            # Get new id (can only use this for tables with autoincrementing ids)
            spsas_id = self.cursor.lastrowid
        else:
            # If they do exist, they will have 1 id, so get that id
            spsas_id = spsas_id[0][0]


        # Check if the feature keys already exist in the table
        feature_keys_id = self.select_feature_keys(params.n_qubits, params.training_feature_keys)
        # If the feature keys do not exist in the table, add them and get the new id
        if not feature_keys_id:
            message = "INSERT INTO feature_keys (n_qubits, "

            # Add a ? for each feature key
            for i in range(1, len(params.training_feature_keys) + 1):
                message += f"feature{i}, "
            # Remove the last ", " from the message
            message = message[:-2]

            # First question mark for n_qubits, then one for each feature key
            message += ") VALUES (?, "
            # Add a ? for each feature key
            for i in range(1, len(params.training_feature_keys)):
                message += "?, "
            message += "?)"

            # Execute the message with the n_qubits and feature keys as parameters
            self.cursor.execute(message, (params.n_qubits, *params.training_feature_keys))
            # Get new id
            feature_keys_id = self.cursor.lastrowid
        else:
            # If they do exist, they will have 1 id, so get that id
            feature_keys_id = feature_keys_id[0][0]
        

        # Check if the data sizes already exist in the table
        data_sizes_id = self.select_data_sizes(params.train_data_size, params.valid_data_size, params.test_data_size)
        # If the data sizes do not exist in the table, add them and get the new id
        if not data_sizes_id:
            self.cursor.execute("INSERT INTO data_sizes (train_data_size, valid_data_size, test_data_size) VALUES (?, ?, ?)", 
                                (params.train_data_size, params.valid_data_size, params.test_data_size))
            data_sizes_id = self.cursor.lastrowid
        else:
            # If they do exist, they will have 1 id, so get that id
            data_sizes_id = data_sizes_id[0][0]


        # Check if the misc params already exist in the table
        misc_params_id = self.select_misc_params(params.n_epochs, params.batch_size, params.num_layers, params.obs_text, params.is_local_simulator, params.use_pca, params.seed)
        # If the misc params do not exist in the table, add them and get the new id
        if not misc_params_id:
            self.cursor.execute("INSERT INTO misc_params (n_epochs, batch_size, num_layers, obs, is_local_simulator, use_pca, seed) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                                (params.n_epochs, params.batch_size, params.num_layers, params.obs_text, params.is_local_simulator, params.use_pca, params.seed))
            misc_params_id = self.cursor.lastrowid
        else:
            # If they do exist, they will have 1 id, so get that id
            misc_params_id = misc_params_id[0][0]


        # Add the new circuit entry to the outputs table
        self.cursor.execute("INSERT INTO outputs (spsas_id, feature_keys_id, data_sizes_id, misc_params_id) VALUES (?, ?, ?, ?)",
                            (spsas_id, feature_keys_id, data_sizes_id, misc_params_id))

        # Commit the changes to the database
        self.conn.commit()

        # Return the id of the new entry in outputs (main) table
        # The entry with this id will later be updated with the results of the run
        return self.cursor.lastrowid







    def update_callback(self, circuit_id, new_data):
        """
        This function updates the database with the new data for the circuit with the given circuit_id.

        Args:
            circuit_id (int): The id of the circuit to update the data for.
            new_data (Results): The new data for the circuit.

        Returns:
            None 
        """
        # Add the new valid loss to the database with each entry corresponding to the new id (all entries with this id represent the losses for this run)
        for entry in new_data.valid_loss:
            self.cursor.execute("INSERT INTO valid_loss (circuit_id, valid_loss) VALUES (?, ?)", (circuit_id, entry))

        # Add the new test data to the database with each entry corresponding to the new id (all entries with this id represent the test data for this run)
        for i in range(len(new_data.test_labels)):
            self.cursor.execute("INSERT INTO test_data (circuit_id, test_label, test_prob, test_pred) VALUES (?, ?, ?, ?)", 
                                (circuit_id, int(new_data.test_labels[i]), new_data.test_prob[i], new_data.test_pred[i]))

        # Add all weights during each iteration to the database with each entry corresponding to the new id (all entries with this id represent the weights for this run)
        for epoch in new_data.weights:
            for i in range(len(new_data.weights[epoch])):
                self.cursor.execute("INSERT INTO parameter_weights (circuit_id, epoch, batch, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18) \
                                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (circuit_id, epoch, i+1, *new_data.weights[epoch][i]))

        # Update the outputs table with the new results
        self.cursor.execute("UPDATE outputs SET cost = ?, test_accuracy = ?, test_data_id = ?, valid_loss_id = ?, run_time = ? WHERE id = ?",
                            (new_data.cost, new_data.test_accuracy, circuit_id, circuit_id, new_data.run_time, circuit_id))

        # Commit the changes to the database
        self.conn.commit()





    
    # If you want any data for a circuit with a certain id, call this function. It will pass all data for that circuit to the caller,
    # and the caller can then do whatever they want with it. If you want data that corresponds to certain conditions, use the other
    # get_data function. If you want data corresponding to an id and certain conditions, use this function and then filter the resulting data however you want.
    def get_data_circuit_id(self, circuit_id):
        """
        Retrieves all data from database for a circuit with the given circuit_id.

        Args:
            circuit_id (int): The id of the circuit to retrieve data for.

        Returns:
            Dictionary: The data for the circuit with the given id. If there is no data for the circuit, returns None.
            Dictionary will be in the following format:
            {
                "circuit_id": int,
                "spsas": {
                    "spsa_alpha": float,
                    "spsa_gamma": float,
                    "spsa_c": float,
                    "spsa_A": float,
                    "spsa_a1": float
                },
                "feature_keys": list of feature keys,
                "data_sizes": [train_data_size, valid_data_size, test_data_size],
                "misc_params": {
                    "n_epochs": int,
                    "batch_size": int,
                    "num_layers": int,
                    "obs": str,
                    "is_local_simulator": bool,
                    "use_pca": bool,
                    "seed": int
                },
                "test_probs": list,
                "test_preds": list,
                "test_labels": list,
                "valid_loss": list,
                "cost": float,
                "test_accuracy": float,
                "run_time": float
            }
        """

        self.cursor.execute("SELECT * FROM outputs JOIN spsas ON outputs.spsas_id = spsas.id JOIN feature_keys ON outputs.feature_keys_id = feature_keys.id \
                                      JOIN data_sizes ON outputs.data_sizes_id = data_sizes.id JOIN misc_params ON outputs.misc_params_id = misc_params.id WHERE outputs.id = ?", (circuit_id,))

        data = self.cursor.fetchone()

        if data:
            test_labels = []
            test_probs = []
            test_preds = []
            # Get the test data for the circuit
            self.cursor.execute("SELECT test_label, test_prob, test_pred FROM test_data WHERE circuit_id = ?", (circuit_id,))
            for entry in self.cursor.fetchall():
                test_labels.append(entry[0])
                test_probs.append(entry[1])
                test_preds.append(entry[2])

            # Get the valid loss for the circuit
            self.cursor.execute("SELECT valid_loss FROM valid_loss WHERE circuit_id = ?", (circuit_id,))
            valid_loss = [entry[0] for entry in self.cursor.fetchall()]

            # Return the data for the circuit
            # IMPORTANT: Indices are based on the current structure of the database. If that changes, this will need to be updated
            data = {
                "circuit_id": circuit_id,
                "spsas": {
                    "spsa_alpha": data[12],
                    "spsa_gamma": data[13],
                    "spsa_c": data[14],
                    "spsa_A": data[15],
                    "spsa_a1": data[16]
                },
                "feature_keys": [data[i] for i in range(19, 19 + data[18])],
                "data_sizes": [data[45], data[46], data[47]],
                "misc_params": {
                    "n_epochs": data[49],
                    "batch_size": data[50],
                    "num_layers": data[51],
                    "obs": data[52],
                    "is_local_simulator": data[53],
                    "use_pca": data[54],
                    "seed": data[55]
                },
                "test_probs": test_probs,
                "test_preds": test_preds,
                "test_labels": test_labels,
                "valid_loss": valid_loss,
                "cost": data[7],
                "test_accuracy": data[8],
                "run_time": data[9]
            }

        return data
        







    def id_message(self, message, ids, params):
        """
        Adds new id query and corresponding question marks to query if ids are not None.
        Updates the parameters list with the ids if they are not None. 

        Args:
            message (str): The message to add the ids to.
            ids (list): The ids to add to the message.
            params (list): The list of parameters to update with the ids.

        Returns:
            str: The updated message.
            list: The updated list of parameters.
        """
        final_message = ""

        if ids:
            for idee in ids:
                message += "?, "
                params.append(idee[0])
            message = message[:-2] + ") AND "

            final_message = message

        return final_message, params



    def process_requests(self):
        # Check for new updates in the queue and process them
        try:
            while True:
                # Try to get an update from the queue, but don't block
                print("Checking for requests")
                (request_x, request_y) = self.request_queue.get(block=False)
                print("Request recieved, gathering data")

                x_data = self.get_data_circuit_id(1)[request_x]
                y_data = range(1, len(x_data))
                # Put the data in the queue 
                self.output_queue.put((request_x, request_y, x_data, y_data))
                print("Added data to queue")
                self.request_queue.task_done()
        except queue.Empty:
            pass

        time.sleep(1)
        self.process_requests()



    # Given a circuit_id a name of a key in the database, and CONDITIONS that the data must pass, return an array of that data. 
    # (MIGHT HAVE TO CHANGE, SINCE THERE MIGHT BE TOO MUCH VARIABILITY IN THIS?)

    # Ideally, we would have some way to pass over conditons that must be met for the data to be included in the results passed back, EX
    # Conditions = order most recent results, first 100 results, results must not == 0 
    def get_conditional_data(self, spsas=None, feature_keys=None, data_sizes=None, misc_params=None, test_accuracy_gt=None, test_accuracy_lt=None, time=None):
        """
        This function retrieves data from the database based on the given conditions. Returns a list of data for the circuits that meet the conditions.
        Usage: If you want to filter by a parameter, pass the parameter. If you don't want to filter by a parameter, pass None for that parameter.
        For simplicity, use parameter=parameter_value in the call for all parameters you want to filter by.

        Args:
            NOTE: If dicts are given for spsas or misc_params, all parameters must be given. To not filter by a parameter, pass None for that parameter.
            Ex: spsas = {"spsa_alpha": None, "spsa_gamma": 0.166, "spsa_c": None, "spsa_A": None, "spsa_a1": 0.1} would query for all entries with spsa_gamma = 0.166 and spsa_a1 = 0.1

            spsas (dict): The spsa parameters to filter the data by. Ex: {"spsa_alpha": 0.1, "spsa_gamma": 0.166, "spsa_c": 0.1, "spsa_A": 3, "spsa_a1": 0.1}
            feature_keys (list): The feature keys to filter the data by. Ex: ["feature1", "feature2", "feature3"]
            data_sizes (list): The data sizes to filter the data by. Ex: [100, 100, 100]
            misc_params (dict): The miscellaneous parameters to filter the data by. Ex: {"n_epochs": 100, "batch_size": 100, "num_layers": 3, "obs": "X", "is_local_simulator": 1, "use_pca": 0, "seed": 100}
            test_accuracy_gt (float): The minimum test accuracy to filter the data by. 
            test_accuracy_lt (float): The maximum test accuracy to filter the data by.
            time (float): The minimum time to filter the data by. TODO

        Returns:
            list: The data for the circuits that meet the conditions. If no circuits meet the conditions, returns an empty list.
            List will be in the same format as the data returned by get_data_circuit_id().
        """
        message = "SELECT id FROM outputs WHERE "
        params = []

        # For each parameter, add the ids of the entries that meet the conditions to the message (if the parameter was given)
        if spsas:
            spsa_ids = self.select_spsas(spsa_alpha=spsas["spsa_alpha"], spsa_gamma=spsas["spsa_gamma"], spsa_c=spsas["spsa_c"], spsa_A=spsas["spsa_A"], spsa_a1=spsas["spsa_a1"])
            new_message, params = self.id_message("spsas_id IN (", spsa_ids, params)
            message += new_message

        if feature_keys:
            feature_keys_ids = self.select_feature_keys(len(feature_keys), feature_keys)
            new_message, params = self.id_message("feature_keys_id IN (", feature_keys_ids, params)
            message += new_message

        if data_sizes:
            data_sizes_ids = self.select_data_sizes(data_sizes[0], data_sizes[1], data_sizes[2])
            new_message, params = self.id_message("data_sizes_id IN (", data_sizes_ids, params)
            message += new_message

        if misc_params:
            misc_params_ids = self.select_misc_params(misc_params["n_epochs"], misc_params["batch_size"], misc_params["num_layers"], misc_params["obs"], 
                                                      misc_params["is_local_simulator"], misc_params["use_pca"], misc_params["seed"])
            new_message, params = self.id_message("misc_params_id IN (", misc_params_ids, params)
            message += new_message

        if test_accuracy_gt:
            test_accuracy_ids = self.cursor.execute("SELECT id FROM outputs WHERE test_accuracy > ?", (test_accuracy_gt,)).fetchall()
            new_message, params = self.id_message("id IN (", test_accuracy_ids, params)
            message += new_message

        if test_accuracy_lt:
            test_accuracy_ids = self.cursor.execute("SELECT id FROM outputs WHERE test_accuracy < ?", (test_accuracy_lt,)).fetchall()
            new_message, params = self.id_message("id IN (", test_accuracy_ids, params)
            message += new_message

        # Don't have time for this now, can implement but will do later
        # All entries where time is greater than the given time
        # if time:
            # time_ids = self.cursor.execute("SELECT id FROM outputs WHERE time > ?", (time,)).fetchall()
            # new_message, params = self.id_message("id IN (", time_ids, params)
            # message += new_message
        
        message = message[:-5]

        ids = []
        # If no parameters were given, don't run
        # Else run the message and get the ids of the entries that meet the conditions
        if len(message) > 28:
            self.cursor.execute(message, (*params,))
            ids = self.cursor.fetchall()

        datas = []
        # Get all data for each circuit that meets the conditions
        for idee in ids:
            data = self.get_data_circuit_id(idee[0])
            if data:
                datas.append(data)

        return datas
    
    
    """
    # I dont understand your code XD so this is placeholder...
    def update_data(self, key):
        # Example of adding a data update to the queue from the database side
        update_data = {'x': x_values, 'y': y_values, 'plot_id': 'some_identifier'}
        self.update_queue.put(update_data)


    
    # Creates and returns a single plot, NOT image. (IE some MPL object we can change and update later...)
    # Based on some variables for the X/Y (I WILL PASS A STRING WITH WHAT I WANT TO PLOT, YOU CALL GET DATA)
    # X_DATA, Y_DATA are TUPLES, with the data (condition, key_name)

    # Save this plot as another entry under the circuit_id dictionary, as a tuple of the (plot reference, and partial function)  
    def plot(self, circuit_id, x_data, y_data):
        # Retrieve data using the provided keys (ignoring conditions for simplicity)
        x_values, y_values = self.get_data(x_data[1]), self.get_data(y_data[1])
        
        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values)
        ax.set(xlabel=x_data[1], ylabel=y_data[1], title=f'Plot for {circuit_id}')
        
        # Store the plot and its update function in the dictionary
        self.plots_dict[circuit_id] = (fig, ax, partial(self.update_XY, circuit_id))
        
        # partial(get_data, 'yes')
        update_queue.put(new_figure)
        return fig, ax

    # Given a list of circuit IDs and a variables for the X/Y list of (CONDITION, STRING) type, generate a plot for each circuit, return list of PLOTS
    def create_plots(self, ids, xs, ys):
        plots = []
        for id, x, y in zip(ids, xs, ys):
            plot = self.plot(id, x, y)
            plots.append(plot)
        return plots

    # Given a plot, UPDATE the X/Y values based on new data
    def update_XY(self, circuit_id, x_data, y_data):
        # Assuming circuit_id exists in plots_dict and has been previously created by plot method
        if circuit_id in self.plots_dict:
            fig, ax, _ = self.plots_dict[circuit_id]
            
            # Clear the current axes and plot new data
            ax.clear()
            x_values, y_values = self.get_data(x_data[1]), self.get_data(y_data[1])
            ax.plot(x_values, y_values)
            ax.set(xlabel=x_data[1], ylabel=y_data[1], title=f'Updated Plot for {circuit_id}')
            fig.canvas.draw()
        else:
            print(f"No plot found for circuit ID {circuit_id}")
    """
    
    # Given a list of circuit IDs, update each plots data corresponding to that circuit ID
    def update_plots(self, ids):
        pass

    # Given a plot, generate a single image with some helpful result name, and return its filepath
    def generate_image(self):
        pass
        

    # Given a plot, generate a GIF of the change rel




    # NOTE: OUTPUT GRAPH outputs/run_id.png WILL BE OVERWRITTEN EVERY TIME THIS FUNCTION IS CALLED
    def compile_run_plots(self, create_data_plot):
        """
        This function compiles the plots for a run into a single document.

        Args:
            run_id (str): The id of the run to compile the plots for.
            save_folder (str): The folder to save the compiled plots to.
        """
        if create_data_plot:
            image_paths = [self.id + "dataplot.png", "validation_loss.png", "classhist.png", "roc.png", "confusion_matrix.png"]
        else:
            image_paths = ["validation_loss.png", "classhist.png", "roc.png", "confusion_matrix.png"]

        images = [Image.open(image_path) for image_path in image_paths]

        # Get aspect ratios for images
        aspect_ratios = [image.width / image.height for image in images]
        # Get new heights for images based on aspect ratios and self.pdf_page_width
        # First image is full width, the rest are half width because they are smaller and therefore paired
        heights = [int(self.pdf_page_width / aspect_ratios[i]) if i == 0 else int((self.pdf_page_width / 2) / aspect_ratios[i]) for i in range(len(aspect_ratios))]

        # Resize images
        resized_images = [image.resize((self.pdf_page_width, heights[i])) if i == 0 else image.resize((int(self.pdf_page_width / 2), heights[i])) for i, image in enumerate(images)]

        # Create new image
        new_image = Image.new("RGB", (self.pdf_page_width, sum(heights[0:3])))

        # Paste images into new image
        x_offset = 0
        y_offset = 0
        # Use offsets to paste images into correct position, first image is full width, the rest are half width
        for i in range(len(resized_images)):
            new_image.paste(resized_images[i], (x_offset, y_offset))
            if i % 2 == 0:
                y_offset += resized_images[i].height
                x_offset = 0
            else:
                x_offset += resized_images[i].width

        # Save new image as run_id.png
        new_image.save(os.path.join(self.savefolder, self.id + ".png"))

        # Remove old images (individual graph pngs)
        for image_path in image_paths:
            os.remove(image_path)

        # So that figures from this run do not stay in memory
        plt.close("all")

        # Could convert to pdf but it blurs the images a bit
        # convert_png_to_pdf(run_id)



    def convert_png_to_pdf(self):
        """
        This function converts a png to a pdf.

        Args:
            run_id (str): The id of the run to convert to a pdf.
        """
        image = Image.open(os.path.join(self.savefolder, self.id + ".png"))
        image.save(os.path.join(self.savefolder, self.id + ".pdf"))
        os.remove(os.path.join(self.savefolder, self.id + ".png"))



    def compile_output_text(self):
        """
        This function moves the text from the results.txt file for a run to outputs/run_id.txt.
        It also adds information about parameters.

        Args:
            run_id (str): The id of the run to get the text for.

        Returns:
            str: The text from the results.txt file for the run.
        """
        with open("results.txt", "r") as f:
            with open(os.path.join(self.savefolder, self.id + ".txt"), "w") as f2:
                f2.write(f"PARAMETERS\n\n")
                f2.write(f"training_feature_keys: {self.params.training_feature_keys}\n")
                f2.write(f"batch_size: {self.params.batch_size}\n")
                f2.write(f"n_epochs: {self.params.n_epochs}\n")
                f2.write(f"use_pca: {self.params.use_pca}\n")
                f2.write(f"train_data_size: {self.params.train_data_size}\n")
                f2.write(f"test_data_size: {self.params.test_data_size}\n")
                f2.write(f"valid_data_size: {self.params.valid_data_size}\n")
                f2.write(f"total_datasize: {self.params.total_datasize}\n")
                f2.write(f"is_local_simulator: {self.params.is_local_simulator}\n")
                f2.write(f"n_qubits: {self.params.n_qubits}\n")
                f2.write(f"num_layers: {self.params.num_layers}\n")
                f2.write(f"obs: {self.params.obs}\n")
                f2.write(f"spsa_alpha: {self.params.spsa_alpha}\n")
                f2.write(f"spsa_gamma: {self.params.spsa_gamma}\n")
                f2.write(f"spsa_c: {self.params.spsa_c}\n")
                f2.write(f"spsa_A: {self.params.spsa_A}\n")
                f2.write(f"spsa_a: {self.params.spsa_a}\n")
                f2.write(f"\n\nRESULTS\n\n")
                f2.write(f.read())
        
        os.remove("results.txt")
    


class Train_Test_Data:

    def __init__(self, params):
        self.params = params

        self.n_signal_events = None
        self.n_background_events = None
        
        self.train_features = None
        self.train_labels = None

        self.valid_features = None
        self.valid_labels = None

        self.rest_features = None
        self.rest_labels = None

        self.test_features = None
        self.test_labels = None


    # TODO: update to use signal / background processes
    def tts_preprocess(self, signal, background):
        # load data from files
        signal_dict, background_dict, files_used = lqm.load_data(
            self.params.signals_folder, self.params.backgrounds_folder, self.params.training_feature_keys
        )

        # formats data for input into vqc
        features, labels = lqm.format_data(signal_dict, background_dict)

        self.n_signal_events = (labels == 1).sum()
        self.n_background_events = (labels == 0).sum()

        features_signal = features[(labels==1)]
        features_background = features[(labels==0)]

        np.random.shuffle(features_signal)
        np.random.shuffle(features_background)

        features = np.concatenate((features_signal[:self.params.half_datasize], features_background[:self.params.half_datasize]))
        # labels = np.array([1]*half_datasize + [0]*half_datasize, requires_grad=False)
        labels = np.array([1]*self.params.half_datasize + [0]*self.params.half_datasize)

        train_features1, rest_features1, self.train_labels, self.rest_labels = train_test_split(
            features,
            labels,
            train_size=self.params.train_data_size,
            test_size=self.params.test_data_size + self.params.valid_data_size,
            random_state=self.params.seed,
            stratify=labels
        )

        # preprocess data (rescaling)
        self.train_features, self.rest_features = lqm.preprocess_data(
            train_features1, rest_features1, self.params.use_pca, self.params.num_features, self.params.seed
        )


        self.valid_features, self.test_features, self.valid_labels, self.test_labels = train_test_split(
            self.rest_features,
            self.rest_labels,
            train_size=self.params.valid_data_size,
            test_size = self.params.test_data_size,
            random_state=self.params.seed,
            stratify=self.rest_labels
        )


