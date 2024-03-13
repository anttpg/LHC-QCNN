import queue
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Interface:
    def __init__(self, master, database, queue):
        # threadsafe queue for the database to give data updates To
        self.queue = queue
        self.db = database
        
        self.left_frame = tk.Frame(master, width=200, height=400)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)
        self.right_frame = tk.Frame(master, width=400, height=400)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)

        self.setup_left_side()
        self.figure_canvas = None


    def setup_left_side(self):
        # Dropdown menus
        dropdown_placeholder = ttk.Combobox(self.left_frame, values=["Option 1", "Option 2", "Option 3"])
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
            check = tk.Checkbutton(self.left_frame, text=f'model{i}', variable=var)
            check.pack(anchor='w')
            self.model_vars.append(var)

    def select_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, file_path)
    

    # USED TO PROCESS UPDATES FROM DB
    def process_updates(self):
        # Check for new updates in the queue and process them
        try:
            while True:
                # Try to get an update from the queue, but don't block
                update_data = self.update_queue.get(block=False)

                plot_id = update_data['plot_id']
                x_values = update_data['x']
                y_values = update_data['y']
                
                # Process the update (e.g., update a graph)
                self.update_or_create_plot(plot_id, x_values, y_values)
                self.queue.task_done()
        except queue.Empty:
            pass

        self.master.after(100, self.process_updates)


    def display_figure(self, figure):
        if self.figure_canvas:
            self.figure_canvas.get_tk_widget().destroy()
        self.figure_canvas = FigureCanvasTkAgg(figure, self.right_frame)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


