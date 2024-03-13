import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('agg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Interface:
    def __init__(self, master, request_queue, update_queue):
        # threadsafe queue for the database to give data updates To
        self.request_queue = request_queue
        self.update_queue = update_queue
        self.master = master
        
        self.left_frame = tk.Frame(master, width=200, height=400)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)
        self.right_frame = tk.Frame(master, width=400, height=400)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)

        self.setup_left_side()
        self.figure_canvas = None
        self.plots = {}  # Store matplotlib figures by plot_id

       
        self.request_queue.put(("valid_loss", "blah"))
        self.master.after(100, self.update)


    def setup_left_side(self):
        # Dropdown menus
        dropdown_placeholder = ttk.Combobox(self.left_frame, values=["VQC Model 1", "VQC Model 2", "VQC Model 3"])
        dropdown_placeholder.pack(pady=5)
        
        # File selection
        self.file_path_entry = tk.Entry(self.left_frame)
        self.file_path_entry.pack(pady=5)
        file_select_button = tk.Button(self.left_frame, text="Select File", command=self.select_file)
        file_select_button.pack(pady=5)
        
        # Table with checkboxes
        self.model_vars = []
        for i in range(1, 6):
            var = tk.BooleanVar()
            check = tk.Checkbutton(self.left_frame, text=f'f_lept_{i}_pt', variable=var)
            check.pack(anchor='w')
            self.model_vars.append(var)
        
        tk.Checkbutton(self.left_frame, text=f'Train hyperparameters?', variable=tk.BooleanVar())
        check.pack(anchor='w')
        

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
                print("Sucessfully recieved data")

                plot_id = "plot1"
                
                # Process the update (e.g., update a graph)
                self.update_or_create_plot(plot_id, x_data, y_data)
                self.update_queue.task_done()
        except queue.Empty:
            print("Requested valid_loss data")
            self.request_queue.put(("valid_loss", "blah"))
            pass
        
        # print("Recursive call to after")
        self.master.after(1400, self.update)


    def update_or_create_plot(self, plot_id, x_values, y_values):
        # Check if the plot already exists, if not create a new figure
        if plot_id not in self.plots:
            self.plots[plot_id] = plt.figure(figsize=(5, 4))
        
        figure = self.plots[plot_id]
        figure.clear()  # Clear the previous plot
        
        # Create the plot
        ax = figure.add_subplot(111)
        ax.plot(x_values, y_values)
        ax.set_title(f"Plot ID: {plot_id}")
        ax.set_xlabel("X Values")
        ax.set_ylabel("Y Values")
        
        # Display the figure using the display_figure method
        self.display_figure(figure)


    def display_figure(self, figure):
        if self.figure_canvas:
            self.figure_canvas.get_tk_widget().destroy()
        self.figure_canvas = FigureCanvasTkAgg(figure, self.right_frame)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

