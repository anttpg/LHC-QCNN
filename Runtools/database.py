from sklearn.model_selection import train_test_split
import Modules.LHC_QML_module as lqm
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt



# Make system that aceepts list of args and plots whichever on a graph against each other


class Train_Test_Data:

        def __init__(self, params, run_id):
            self.params = params
            self.id = run_id
            self.savefolder = "outputs"
            self.pdf_page_width = 650

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



        def plot_datapoints(self):
            signal_dict, background_dict, files_used = lqm.load_data(
                self.params.signals_folder, self.params.backgrounds_folder, self.params.training_feature_keys
            )

            lqm.plot_pairwise_dicts(signal_dict, background_dict)
            plt.savefig(self.id + "dataplot.png")



        # NOTE: OUTPUT GRAPH outputs/run_id.png WILL BE OVERWRITTEN EVERY TIME THIS FUNCTION IS CALLED
        def compile_run_plots(self):
            """
            This function compiles the plots for a run into a single document.

            Args:
                run_id (str): The id of the run to compile the plots for.
                save_folder (str): The folder to save the compiled plots to.
            """
            image_paths = [self.id + "dataplot.png", "validation_loss.png", "classhist.png", "roc.png", "confusion_matrix.png"]

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