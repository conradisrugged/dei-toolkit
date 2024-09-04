import os  # Import the 'os' module for interacting with the operating system
import torch  # Import the 'torch' library for tensor operations and deep learning
from transformers import T5Tokenizer, T5ForConditionalGeneration  # Import classes for the Flan-T5 model and tokenizer
import tkinter as tk  # Import the 'tkinter' library for creating graphical user interfaces (GUI)
from tkinter import messagebox  # Import the 'messagebox' module for displaying message boxes in the GUI
import psutil  # Import the 'psutil' library for retrieving information about system resources
import traceback  # Import the 'traceback' module for printing stack traces of exceptions

def get_device():
    """
    Determines the available computing device (CUDA or CPU) for PyTorch computations.

    Returns:
        torch.device: The available device, either "cuda" if a CUDA-enabled GPU is present, or "cpu" otherwise.
    """
    # Check if CUDA (GPU support) is available using 'torch.cuda.is_available()'.
    if torch.cuda.is_available():
        # If CUDA is available, return a 'torch.device' object representing the CUDA device.
        return torch.device("cuda")
    else:
        # If CUDA is not available, return a 'torch.device' object representing the CPU.
        return torch.device("cpu")

def load_model_and_tokenizer(model_path):
    """
    Loads a pre-trained Flan-T5 model and its tokenizer from the specified directory.

    Args:
        model_path (str): The directory containing the pre-trained Flan-T5 model files.

    Returns:
        tuple: A tuple containing the loaded Flan-T5 model (T5ForConditionalGeneration) and
               the T5Tokenizer, or (None, None) if loading fails.
    """
    # Use a try-except block to handle potential exceptions during model loading
    try:
        # Load the T5Tokenizer from the specified 'model_path'
        tokenizer = T5Tokenizer.from_pretrained(model_path)

        # Load the Flan-T5 model (T5ForConditionalGeneration) from the 'model_path'
        # and move it to the device determined by 'get_device()'.
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(get_device())

        # Return the loaded model and tokenizer
        return model, tokenizer

    except Exception as e:  # Catch any exception that occurs during loading
        # Create a root Tkinter window for displaying the error message
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Display an error message box using 'messagebox.showerror', indicating
        # the failure to load the model and the exception message.
        messagebox.showerror("Error", f"Failed to load model from {model_path}: {str(e)}")
        root.destroy()  # Destroy the root window after the message box is closed

        # Return None for both model and tokenizer to indicate failure
        return None, None

def find_first_valid_model_dir(educated_models_dir):
    """
    Searches for the first valid Flan-T5 model directory within the given root directory.

    Args:
        educated_models_dir (str): The root directory where to search for model directories.

    Returns:
        str: The path to the first valid model directory found, or None if no valid directory is found.
    """
    # Iterate through each item (directory name) in the 'educated_models_dir'
    for dir_name in os.listdir(educated_models_dir):
        # Construct the full path to the current directory
        model_path = os.path.join(educated_models_dir, dir_name)

        # Check if the 'model_path' is a directory and if it's a valid Flan-T5 model directory
        if os.path.isdir(model_path) and is_valid_model_dir(model_path):
            # If it's a valid directory, return the path
            return model_path

    # If no valid directory is found, return None
    return None

def is_valid_model_dir(directory):
    """
    Checks if the given directory contains the necessary files for a valid Flan-T5 model.
    Looks for specific file extensions (.json, .model, .safetensors, .bin) that indicate
    the presence of model configuration, vocabulary, and weights.

    Args:
        directory (str): The directory to check for model files.

    Returns:
        bool: True if the directory contains the necessary files for a valid model, False otherwise.
    """
    # Initialize boolean flags to track the presence of required files
    has_json = False  # Flag for .json file (configuration)
    has_model = False  # Flag for .model file (vocabulary)
    has_safetensors = False  # Flag for .safetensors file (model weights)
    has_bin = False  # Flag for .bin file (model weights)

    # Iterate through each file name in the specified directory
    for filename in os.listdir(directory):
        # Check for specific file extensions and set the corresponding flags if found
        if filename.endswith(".json"):
            has_json = True
        elif filename.endswith(".model"):
            has_model = True
        elif filename.endswith(".safetensors"):
            has_safetensors = True
        elif filename.endswith(".bin"):
            has_bin = True

    # Return True if all necessary file flags are True, indicating a valid model directory
    return has_json and has_model and (has_safetensors or has_bin)

def check_system_memory():
    """
    Retrieves system memory information using the psutil library.

    Returns:
        tuple: A tuple containing two psutil._common.svmem objects representing physical
               and swap memory information, or (None, None) if an error occurs during retrieval.
    """
    try:
        # Get physical memory information using 'psutil.virtual_memory()'
        physical_memory = psutil.virtual_memory()

        # Get swap memory information using 'psutil.swap_memory()'
        swap_memory = psutil.swap_memory()

        # Return the physical and swap memory information as a tuple
        return physical_memory, swap_memory

    except Exception as e:  # Catch any exceptions during memory information retrieval
        # Print an error message and the stack trace to the console
        error_message = f"Error checking system memory: {e}\n{traceback.format_exc()}"
        print(error_message)

        # Return (None, None) to indicate an error occurred
        return None, None

def check_cuda_devices():
    """
    Checks for CUDA availability and the number of CUDA devices.

    Returns:
        torch.device: The default device ("cuda" if CUDA is available, "cpu" otherwise),
                       or None if an error occurs while checking.
    """
    try:
        print("Checking for CUDA devices...")  # Print a message to indicate CUDA check

        # Check if CUDA is available using 'torch.cuda.is_available()'
        if torch.cuda.is_available():
            # If CUDA is available, print the number of GPUs
            print(f"CUDA is available with {torch.cuda.device_count()} GPU(s).")

            # Set the default device to CUDA and print a confirmation message
            cuda_device = torch.device('cuda')
            print(f"Setting default device to CUDA: {cuda_device}")
            torch.cuda.set_device(0)  # Set the first GPU as the default

            # Return the CUDA device
            return cuda_device

        else:
            # If CUDA is not available, print a message indicating CPU usage
            print("CUDA is not available. Using CPU.")

            # Return the CPU device
            return torch.device('cpu')

    except Exception as e:  # Catch any exceptions that occur during CUDA device checks
        # Print the error message and stack trace to the console
        error_message = f"Error checking CUDA devices: {e}\n{traceback.format_exc()}"
        print(error_message)
        return None  # Return None to indicate an error occurred during the check
