#setup
import os

#original directory
orig_input_dataset = "flower_photos"

#new directory after splitting
base_path = "flower"

#the training, validation, and testing directories
train_path = os.path.sep.join([base_path, "training"])
val_path = os.path.sep.join([base_path, "validation"])
test_path = os.path.sep.join([base_path, "testing"])

# splitting ratios
train_split = 0.8
val_split = 0.1
