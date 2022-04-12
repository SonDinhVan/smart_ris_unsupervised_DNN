"""
This script generates data for upsupervised DNN.
"""
from module import util
from module import data_generator as dgr
from module import system
import pickle
import time

start_time = time.time()
# Load the configuration
config = util.load_config("./configs/config.yaml")
path_to_train_data = config["path"]["path_to_train_data"]

# Network topology
network = system.Network(config=config)
# Data Generator
data_generator = dgr.DataGenerator(network)

# Generate training data
num_samples = 500000

print("Generating {} training data samples".format(num_samples))
print("---------------------------------")
print(
    "Number of phases = {} | Number of CHs = {}".format(
        network.RIS.num_phase, network.K
    )
)
train_data = data_generator.generate_training_data(
    num_samples=num_samples, cascaded_data=True, random_angles=False
)

# Write config and training data into a pickle use for further use
with open(path_to_train_data, "wb") as f:
    pickle.dump({"config": config, "data": train_data}, f)
end_time = time.time()
print("---------------------------------")
print("Finished after {: .2f} secs. Data is stored in {}".format(
    end_time - start_time, path_to_train_data))
