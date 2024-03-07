from sklearn.model_selection import train_test_split
import Modules.LHC_QML_module as lqm
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import sqlite3
from functools import partial


class Database:

    def __init__(self, filepath):
        # self.circuit_list = #FIGURE OUT?
        self.conn = sqlite3.connect(filepath)
        self.cursor = self.conn.cursor()



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




    def select_spsas(self, spsa_alpha, spsa_gamma, spsa_c, spsa_A, spsa_a1):
        """
        This function selects the id of the given spsa parameters from the database. Or None if the parameters do not exist. 
        """
        self.cursor.execute("SELECT id FROM spsas WHERE spsa_alpha = ? AND spsa_gamma = ? AND spsa_c = ? AND spsa_A = ? AND spsa_a1 = ?", 
                            (spsa_alpha, spsa_gamma, spsa_c, spsa_A, spsa_a1))
        # Will be none if the parameters do not exist, otherwise will be the id of the parameters inside a list
        spsa_id = self.cursor.fetchone()
        return spsa_id if spsa_id is None else spsa_id[0]





    def select_feature_keys(self, n_qubits, feature_keys):
        """
        This function selects the id of the given feature keys from the database. Or None if the keys do not exist. 
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
        feature_keys_id = self.cursor.fetchone()
        return feature_keys_id if feature_keys_id is None else feature_keys_id[0]




    def select_data_sizes(self, train_data_size, valid_data_size, test_data_size):
        """
        This function selects the id of the given data sizes from the database. Or None if the sizes do not exist. 
        """
        self.cursor.execute("SELECT id FROM data_sizes WHERE train_data_size = ? AND valid_data_size = ? AND test_data_size = ?", 
                            (train_data_size, valid_data_size, test_data_size))
        # Will be none if the parameters do not exist, otherwise will be the id of the parameters inside a list
        data_sizes_id = self.cursor.fetchone()
        return data_sizes_id if data_sizes_id is None else data_sizes_id[0]




    def select_misc_params(self, n_epochs, batch_size, num_layers, obs, is_local_simulator, use_pca, seed):
        """
        This function selects the id of the given misc parameters from the database. Or None if the parameters do not exist. 
        """
        self.cursor.execute("SELECT id FROM misc_params WHERE n_epochs = ? AND batch_size = ? AND num_layers = ? AND obs = ? AND is_local_simulator = ? AND use_pca = ? AND seed = ?", 
                            (n_epochs, batch_size, num_layers, obs, is_local_simulator, use_pca, seed))
        # Will be none if the parameters do not exist, otherwise will be the id of the parameters inside a list
        misc_params_id = self.cursor.fetchone()
        return misc_params_id if misc_params_id is None else misc_params_id[0]





    def create_circuit_entry(self, params):
        """
        Create a new entry in the circuits table with the given parameters. Return the id of the new entry as the unique id for the current circuit.
        """

        # Check if the spsa params already exist in the table
        spsas_id = self.select_spsas(params.spsa_alpha, params.spsa_gamma, params.spsa_c, params.spsa_A, params.spsa_a1)
        # If the spsa params do not exist in the table, add them and get the new id
        if spsas_id is None:
            self.cursor.execute("INSERT INTO spsas (spsa_alpha, spsa_gamma, spsa_c, spsa_A, spsa_a1) VALUES (?, ?, ?, ?, ?)", 
                                (params.spsa_alpha, params.spsa_gamma, params.spsa_c, params.spsa_A, params.spsa_a1))
            # Get new id (can only use this for tables with autoincrementing ids)
            spsas_id = self.cursor.lastrowid


        # Check if the feature keys already exist in the table
        feature_keys_id = self.select_feature_keys(params.n_qubits, params.training_feature_keys)
        # If the feature keys do not exist in the table, add them and get the new id
        if feature_keys_id is None:
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
        

        # Check if the data sizes already exist in the table
        data_sizes_id = self.select_data_sizes(params.train_data_size, params.valid_data_size, params.test_data_size)
        # If the data sizes do not exist in the table, add them and get the new id
        if data_sizes_id is None:
            self.cursor.execute("INSERT INTO data_sizes (train_data_size, valid_data_size, test_data_size) VALUES (?, ?, ?)", 
                                (params.train_data_size, params.valid_data_size, params.test_data_size))
            data_sizes_id = self.cursor.lastrowid


        # Check if the misc params already exist in the table
        misc_params_id = self.select_misc_params(params.n_epochs, params.batch_size, params.num_layers, params.obs_text, params.is_local_simulator, params.use_pca, params.seed)
        # If the misc params do not exist in the table, add them and get the new id
        if misc_params_id is None:
            self.cursor.execute("INSERT INTO misc_params (n_epochs, batch_size, num_layers, obs, is_local_simulator, use_pca, seed) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                                (params.n_epochs, params.batch_size, params.num_layers, params.obs_text, params.is_local_simulator, params.use_pca, params.seed))
            misc_params_id = self.cursor.lastrowid


        # Add the new circuit entry to the outputs table
        self.cursor.execute("INSERT INTO outputs (spsas_id, feature_keys_id, data_sizes_id, misc_params_id) VALUES (?, ?, ?, ?)",
                            (spsas_id, feature_keys_id, data_sizes_id, misc_params_id))

        # Commit the changes to the database
        self.conn.commit()

        # Return the id of the new entry in outputs (main) table
        # The entry with this id will later be updated with the results of the run
        return self.cursor.lastrowid





    # Callback function to be called by controller every time it completes a runner (ie like run_one completes), signal to the database that new 
    # Data has appeared, allowing us to update the graphs, AND  SOMEHOW SAVE DATA
    def update_callback(self, circuit_id, new_data):
        # Add the new valid loss to the database with each entry corresponding to the new id (all entries with this id represent the losses for this run)
        for entry in new_data.valid_loss:
            self.cursor.execute("INSERT INTO valid_loss (circuit_id, valid_loss) VALUES (?, ?)", (circuit_id, entry))

        # Add the new test data to the database with each entry corresponding to the new id (all entries with this id represent the test data for this run)
        for i in range(len(new_data.test_labels)):
            self.cursor.execute("INSERT INTO test_data (circuit_id, test_label, test_prob, test_pred) VALUES (?, ?, ?, ?)", 
                                (circuit_id, int(new_data.test_labels[i]), new_data.test_prob[i], new_data.test_pred[i]))

        # Update the outputs table with the new results
        self.cursor.execute("UPDATE outputs SET cost = ?, test_accuracy = ?, test_data_id = ?, valid_loss_id = ?, run_time = ? WHERE id = ?",
                            (new_data.cost, new_data.test_accuracy, circuit_id, circuit_id, new_data.run_time, circuit_id))

        # Commit the changes to the database
        self.conn.commit()





    # Given a circuit_id a name of a key in the database, and CONDITIONS that the data must pass, return an array of that data. 
    # (MIGHT HAVE TO CHANGE, SINCE THERE MIGHT BE TOO MUCH VARIABILITY IN THIS?)

    # Ideally, we would have some way to pass over conditons that must be met for the data to be included in the results passed back, EX
    # Conditions = order most recent results, first 100 results, results must not == 0 
    def get_data(self, circuit_id, conditions, key_name):
        return None
    

    # Creates and returns a single plot, NOT image. (IE some MPL object we can change and update later...)
    # Based on some variables for the X/Y (I WILL PASS A STRING WITH WHAT I WANT TO PLOT, YOU CALL GET DATA)
    # X_DATA, Y_DATA are TUPLES, with the data (condition, key_name)

    # Save this plot as another entry under the circuit_id dictionary, as a tuple of the (plot reference, and partial function)
    def plot(self, circuit_id, x_data, y_data):

        # partial(get_data, 'yes')
        return None
    
    # Given a list of circuit IDs and a variables for the X/Y list of (CONDITION, STRING) type, generate a plot for each circuit, return list of PLOTS
    def create_plots(self, ids, xs, ys):
        return None
    
    # Given a plot, UPDATE the X/Y values based on new data
    def update_XY(self, plot, x_data, y_data):
        return None
    
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


