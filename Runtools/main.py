from controller import Controller
#from multiprocessing import Pool TODO Implement multiprocessing 

database = Database() # Feed reference to controller 1
test_c = Controller(database)
 
test_c.create_runner(None, None, "1") # Give specific hyperparameters, otherwise use some defaults 3
test_c.create_runner(None, None, "2")
test_c.create_runner(None, None, "3")
test_c.run_all() # Figure out why its erroring

database.graph(____) # Show basic visualization in MPL 2