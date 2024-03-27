import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('agg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Interface:
    def __init__(self, master, request_queue, update_queue, user_queue):
        # threadsafe queue for the database to give data updates To
        self.request_queue = request_queue
        self.update_queue = update_queue
        self.master = master
        self.user_queue = user_queue
        
        self.left_frame = tk.Frame(master, width=200, height=400)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)

        self.left_sub_frame = tk.Frame(self.left_frame) 
        self.right_sub_frame = tk.Frame(self.left_frame, bg='lightblue')  
        
        #  sub-frames within the left frame get horizontal layout
        self.left_sub_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_sub_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


        self.right_frame = tk.Frame(master, width=400, height=400)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)

        # TODO move elsewhere in the code??
        self.training_feature_keys = ['f_lept3_pt', 'f_lept4_pt', 'f_Z1mass', 'f_angle_costheta2', 'f_pt4l', 'f_eta4l', 'f_jet1_pt', 'f_jet1_e']
        self.models = ["VQC Model 1", "VQC Model 2", "VQC Model 3"]
        self.setup_left_side()

        self.figure_canvas = None
        self.plots = {}  # Store matplotlib figures by plot_id
        self.figure_canvas = []  # store canvas objects
        self.max_plots = 4
        self.grid_size = 2
        

        # Start waiting for input
        self.master.after(100, self.update)


    def setup_left_side(self):
        # VQC Selection
        header_label = tk.Label(self.left_sub_frame, text="Select VQC model to use")
        header_label.pack(anchor='w')
        dropdown_placeholder = ttk.Combobox(self.left_sub_frame, values=self.models)
        dropdown_placeholder.pack(pady=5)
        
        # Feature Key Selection
        header_label = tk.Label(self.left_sub_frame, text="Select feature keys to run")
        header_label.pack(anchor='w')

        self.model_vars = []
        for key in self.training_feature_keys:
            var = tk.BooleanVar()
            check = tk.Checkbutton(self.left_sub_frame, text=key, variable=var)
            check.pack(anchor='w')
            self.model_vars.append(var)
        
        # Number of iterations to train 
        header_label = tk.Label(self.left_sub_frame, text="Number of iterations to train")
        header_label.pack(anchor='w')
        self.iterations_entry = tk.Entry(self.left_sub_frame)
        self.iterations_entry.pack(pady=5)

        train_hyperparameters_check = tk.Checkbutton(self.left_sub_frame, text='Train hyperparameters?', variable=tk.BooleanVar())
        train_hyperparameters_check.pack(anchor='w')

        # Load from previous run?
        header_label = tk.Label(self.left_sub_frame, text="Load from previous run?")
        header_label.pack(anchor='w')
        file_select_button = tk.Button(self.left_sub_frame, text="Select File", command=self.select_file)
        file_select_button.pack(pady=5)



    
        # Graph generator
        circuit_dropdown = ttk.Combobox(self.right_sub_frame, values=[1, 2, 3, 4])
        circuit_dropdown.pack(pady=5)

        # Data type selection
        data_type_dropdown = ttk.Combobox(self.right_sub_frame, values=["valid_loss", "cost"])
        data_type_dropdown.pack(pady=5)

        # Graph generator
        compare_button = tk.Button(self.right_sub_frame, text="Compare", command=lambda: self.request_data(circuit_dropdown.get(), data_type_dropdown.get()))
        compare_button.pack(pady=5, padx=20)


    # Requests data be sent to update (For graph generation)
    def request_data(self, circuit_id, data_type):
        print(f"Requesting {data_type} data for circuit {circuit_id}")
        self.request_queue.put((data_type, int(circuit_id)))

    # Select a file dialouge box
    def select_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, file_path)


    # USED TO PROCESS UPDATES FROM DB
    def update(self):
        # Check for new updates in the queue and process them
        try:
            while True:
                # Try to get an update from the queue, but don't block
                #print("Checking for new data")
                (request_x, request_y, x_data, y_data) = self.update_queue.get(block=False)
                print("Sucessfully recieved " + str(request_x) + " data")

                plot_id = f"plot {request_y}, {request_x} "
                
                # Process the update (e.g., update a graph)
                # print(x_data)
                # print(y_data)
                self.update_or_create_plot(plot_id, x_data, y_data)
                self.update_queue.task_done()
        except queue.Empty:
            # print("Requesting valid_loss data")
            # for circuit_id in range(1, 2):
            #     self.request_queue.put(("valid_loss", circuit_id))
            pass
        
        # print("Recursive call to after")
        self.master.after(1400, self.update)


    def update_or_create_plot(self, plot_id, x_values, y_values):
        # Check if the plot already exists, if not create a new figure

        if plot_id not in self.plots:
            # If we reached the max number of plots, remove oldest one
            if len(self.plots) >= self.max_plots:
                oldest_plot_id = next(iter(self.plots))
                del self.plots[oldest_plot_id]
                self.clear_canvas(oldest_plot_id)
            
            self.plots[plot_id] = plt.figure(figsize=(5, 4))
        
        figure = self.plots[plot_id]
        figure.clear()  # Clear the previous plot

        # Create the plot
        ax = figure.add_subplot(111)
        ax.plot(x_values, y_values)
        ax.set_title(f"Plot ID: {plot_id}")
        ax.set_xlabel("X Values")
        ax.set_ylabel("Y Values")

        # Display all figures
        self.display_all_figures()

    # TODO maybe?
    def clear_canvas(self, plot_id):
        pass

    def display_all_figures(self):
        # clear all existing canvases
        for canvas in self.figure_canvas:
            canvas.get_tk_widget().destroy()
        self.figure_canvas.clear()

        # create a new canvas for each figure and arrange them in a grid
        for index, (plot_id, figure) in enumerate(self.plots.items()):
            row = index // self.grid_size
            column = index % self.grid_size
            canvas = FigureCanvasTkAgg(figure, self.right_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=row, column=column, sticky="nsew")
            self.right_frame.grid_rowconfigure(row, weight=1)
            self.right_frame.grid_columnconfigure(column, weight=1)
            self.figure_canvas.append(canvas)