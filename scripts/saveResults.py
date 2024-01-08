#  create a directory named based on batch size, learning rate, and loss (0 or 1), and inside it, generate a sub-directory with the current date and time.
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import os

def createResultsDir(batch_size,lr,loss):
    # Get current date and time
    current_datetime = datetime.now()

    # Format the current date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Create directory name based on batch size, learning rate, and loss
    directory_name = f"batch_{batch_size}_lr_{lr}_loss_{loss}"

    # Create a directory with the formatted directory name
    outer_directory_path = os.path.join("results", directory_name)

    try:
        # Check if the directory exists, if not, create it
        if not os.path.exists(outer_directory_path):
            # Create the outer directory
            os.mkdir(outer_directory_path)
        print(f"Outer Directory '{directory_name}' created successfully.")

        # Inside the outer directory, create a directory with the formatted datetime
        inner_directory_path = os.path.join(outer_directory_path, formatted_datetime)
        os.mkdir(inner_directory_path)

        print(f"Inner Directory '{formatted_datetime}' created successfully in '{directory_name}'.")
    except OSError:
        print(f"Creation of directories failed.")
    return inner_directory_path

# save results in CSV
def SaveResults(result_dir, batch_size, num_epochs, lr, MSE, RMSE):
    # Specify the CSV file path
    csv_file_path = os.path.join(result_dir, 'results.csv')
    print(csv_file_path)
    # Writing an empty CSV file if it doesn't exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)

    # Write data to CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:
        # Define the header names
        fieldnames = ['Batch', 'Epoch', 'LR', 'MSE', 'RMSE']
        
        # Create a CSV writer object with specified fieldnames
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Check if the file is empty, if so, write the header row
        if csvfile.tell() == 0:
            csvwriter.writeheader()
            
        # Write the data row
        csvwriter.writerow({'Batch': batch_size, 'Epoch': num_epochs, 'LR': lr, 'MSE': MSE, 'RMSE': RMSE})


def saveLossDiagram(num_epochs,losses,result_dir):
    # Create a list of indices (epochs)
    epochs = list(range(0, num_epochs))
    # Plotting the graph
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses , marker='o', color='#0802A3', label='Loss')
    plt.ylabel('Loss',fontsize=15)
    plt.xlabel('Epoch',fontsize=15)
    # Increase Xticks and Yticks Size
    plt.xticks(fontsize=15)  # Set the fontsize for xticks
    plt.yticks(fontsize=15)  # Set the fontsize for yticks

    plt.title('Model-Loss')
    # plt.grid(True)
    plt.legend(fontsize=25)
    plt.savefig(result_dir+'/model_loss.png')
    plt.show()


def observedvsPredicted(result_dir, targets, outputs):
    t_len =int(len(targets)/200) 
    o_len = int(len(outputs)/200)

    # Extract the values from the tensors
    # Create a list of index positions (x-values)
    indices_t = list(range(t_len))  
    indices_o = list(range(o_len))

    # Create a plot
    plt.figure(figsize=(40, 10))  # Adjust the figure size as needed

    # Plot the values against their index positions
    plt.plot(indices_t, targets[0:t_len], marker='o', linestyle='-', color='blue')
    plt.plot(indices_o, outputs[0:o_len], marker='o', linestyle='-', color='red')

    # Add labels and a title
    plt.ylabel('Value', fontsize=25)  # Adjust fontsize for ylabel
    plt.title('Observed and predicted values', fontsize=25)  # Adjust fontsize for title

    # Increase Legend Size
    plt.legend(labels=['Observed', 'Predicted'], fontsize=25)
    plt.subplots_adjust(bottom=0.15)
    # Increase Xticks and Yticks Size
    plt.xticks(fontsize=25)  # Set the fontsize for xticks
    plt.yticks(fontsize=25)  # Set the fontsize for yticks

    # Save and display the plot
    plt.savefig(os.path.join(result_dir,'performance.png'))
    plt.show()
