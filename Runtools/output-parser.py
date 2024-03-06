import os

# Define the threshold for test accuracy
threshold = 0.85

# Define a function to extract test accuracy from a line of text
def extract_test_accuracy(line):
    test_accuracy = 0
    if "Test accuracy:" in line:
        # Split the line by '|'
        parts = line.split('|')
        # Extract the test accuracy
        test_accuracy_str = parts[1].strip().split(':')[-1].strip()
        # Convert test accuracy to float
        test_accuracy = float(test_accuracy_str)
        return test_accuracy

# Directory containing .txt files
directory = "./outputs"

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        # Open and read the file
        with open(filepath, "r") as file:
            # Check each line for test accuracy
            for line in file:
                if "Test accuracy:" in line:
                    # Extract test accuracy
                    test_accuracy = extract_test_accuracy(line)
                    # Check if test accuracy exceeds the threshold
                    if test_accuracy > threshold:
                        print(f"File '{filename}' has test accuracy above {threshold}: {test_accuracy}")
                        # If you want to stop checking further lines in the file after finding the accuracy
                        # break

