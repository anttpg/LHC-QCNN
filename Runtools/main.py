from controller import Controller
#from multiprocessing import Pool TODO Implement multiprocessing 

# database = Database() # Feed reference to controller 1
test_c = Controller()
 
test_c.create_runner(None, None, "1") # Give specific hyperparameters, otherwise use some defaults 2
test_c.create_runner(None, None, "2")
test_c.create_runner(None, None, "3")
test_c.run_all()