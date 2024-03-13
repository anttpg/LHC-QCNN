import queue
import tkinter as tk
import threading
from matplotlib.figure import Figure
import numpy as np
import time

# Assuming Interface class is defined as in the previous message

def simulate_database_updates(update_queue):
    """
    Simulates sending data updates from the database to the interface.
    This function runs in its own thread to simulate asynchronous updates.
    """
    plot_ids = ['plot1', 'plot2']  # Example plot IDs for updates

    for _ in range(10):  # Simulate 10 rounds of updates
        for plot_id in plot_ids:
            # Generate some example data updates
            x_values = np.arange(0, 5, 0.1)
            y_values = np.sin(x_values) + np.random.normal(0, 0.1, len(x_values))

            # Construct the update data packet
            update_data = {'plot_id': plot_id, 'x': list(x_values), 'y': list(y_values)}
            
            # Send the update to the queue
            update_queue.put(update_data)

            # Wait a bit before sending the next update
            time.sleep(1)

def main_test():
    update_queue = queue.Queue()

    # Setup and run the interface in its own thread
    def run_interface():
        root = tk.Tk()
        app = Interface(root, None, update_queue)  # Database is set to None for this test
        root.after(100, app.process_updates)  # Start processing updates after setup
        root.mainloop()

    interface_thread = threading.Thread(target=run_interface, args=())
    interface_thread.start()

    # Simulate database updates in a separate thread
    db_simulation_thread = threading.Thread(target=simulate_database_updates, args=(update_queue,))
    db_simulation_thread.start()

if __name__ == "__main__":
    main_test()
