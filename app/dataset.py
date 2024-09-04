import torch  # Import the PyTorch library for tensor operations and deep learning
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader for creating and managing datasets
import pandas as pd  # Import pandas library for data analysis and manipulation, especially with DataFrames
import re  # Import the 're' module for regular expression operations

# Define the CustomDataset class, which inherits from the PyTorch Dataset class
class CustomDataset(Dataset):
    """
    Custom dataset class to handle loading and preprocessing of training data.
    Inherits from the PyTorch Dataset class to ensure compatibility with PyTorch's
    data loading mechanisms.

    This class takes training data (typically pairs of prompts and responses),
    a tokenizer for processing text, a maximum sequence length for padding/truncation,
    and a flag indicating whether the dataset is used for evaluation.
    """
    def __init__(self, data, tokenizer, max_length=128, eval_mode=False):
        """
        Initializes the CustomDataset instance with the provided data and parameters.

        Args:
            data (list): The training data, which can be:
                         - A list of (prompt, response) pairs: [["prompt1", "response1"], ["prompt2", "response2"], ...]
                         - A list of prompts only: ["prompt1", "prompt2", ...]
            tokenizer (transformers.T5Tokenizer): The T5 tokenizer used for encoding text data.
            max_length (int, optional): The maximum sequence length for padding and truncation.
                                        Defaults to 128.
            eval_mode (bool, optional): A flag indicating whether the dataset is used for evaluation.
                                        Defaults to False.
        """
        self.data = data  # Store the training data
        self.tokenizer = tokenizer  # Store the provided tokenizer
        self.max_length = max_length  # Store the maximum sequence length
        self.eval_mode = eval_mode  # Store the evaluation mode flag

    def __len__(self):
        """
        Returns the number of data items in the dataset.

        Returns:
            int: The length of the dataset (number of prompts or prompt-response pairs).
        """
        return len(self.data)  # Return the length of the data list

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses a data item at the specified index.

        Args:
            idx (int): The index of the data item to retrieve.

        Returns:
            dict: A dictionary containing the preprocessed input tensors for the Flan-T5 model:
                  - 'input_ids': Tensor of input token IDs
                  - 'attention_mask': Tensor indicating valid input tokens
                  - 'labels': Tensor of target token IDs (for training)

        Raises:
            IndexError: If the provided index is out of range for the dataset.
        """
        if idx >= len(self.data):
            raise IndexError("Index out of range")  # Raise an error if the index is invalid

        data_item = self.data[idx]  # Get the data item at the given index

        # Check if the data item is a list with two elements (prompt and response)
        if isinstance(data_item, list) and len(data_item) == 2:
            prompt, response = data_item  # Unpack the prompt and response

            # Ensure the response is a string; join list elements if it's a list
            if isinstance(response, list):
                response = " ".join(response)
            else:
                response = str(response)

        # If the data item is not a list of two elements, assume it's only a prompt
        else:
            prompt = data_item  # The data item is the prompt
            response = ""  # Set an empty response for single prompt inputs

        # Encode the prompt and response using the tokenizer
        encoding = self.tokenizer(
            prompt,  # The prompt text
            text_target=response,  # The response text (empty if only a prompt is provided)
            max_length=self.max_length,  # Maximum sequence length
            padding="max_length",  # Pad sequences to the maximum length
            truncation=True,  # Truncate sequences longer than the maximum length
            return_tensors="pt"  # Return PyTorch tensors
        )

        # Create labels for training by copying the input IDs
        labels = encoding['input_ids'].clone()

        # Set labels for padded tokens to -100, indicating they should be ignored during loss calculation
        labels[encoding['attention_mask'] == 0] = -100

        # Create a dictionary to store the input tensors for the model
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),  # Input token IDs
            'attention_mask': encoding['attention_mask'].flatten(),  # Attention mask
            'labels': labels.flatten()  # Labels for training
        }

        return inputs  # Return the preprocessed input tensors

def clean_name(name):
    """
    Removes quotes and commas from a name string to ensure it can be used safely in file paths.

    Args:
        name (str): The name string to clean.

    Returns:
        str: The cleaned name string without quotes and commas.
    """
    return re.sub(r'["\',]', "", name)  # Use a regular expression to remove quotes and commas

def load_dataset(path, person_name="Conrad", model_name="Logos"):
    """
    Loads a dataset from a CSV file and personalizes it with provided names.
    The CSV file should have two columns: "input" and "output", representing
    prompts and responses respectively.

    Args:
        path (str): The path to the CSV file containing the dataset.
        person_name (str, optional): The name to replace "Conrad" with in the dataset.
                                    Defaults to "Conrad".
        model_name (str, optional): The name to replace "Logos" with in the dataset.
                                    Defaults to "Logos".

    Returns:
        list: A list of lists, where each inner list represents a data item:
              - [prompt, response] for prompt-response pairs.
              - [prompt, "[No Response]"] if the response is missing or empty.
              - None if an error occurs during loading.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(path, names=["input", "output"])

        # Clean the provided names
        person_name = clean_name(person_name)
        model_name = clean_name(model_name)

        # Iterate through each row of the DataFrame
        for row in df.itertuples():
            # Replace "Conrad" with 'person_name' in the "input" column if "Conrad" is present
            if 'Conrad' in row.input:
                df.at[row.Index, "input"] = row.input.replace("Conrad", person_name)
            # Replace "Logos" with 'model_name' in the "input" column if "Logos" is present
            if 'Logos' in row.input:
                df.at[row.Index, "input"] = row.input.replace("Logos", model_name)

            # If the "output" is NaN or an empty string, set it to "[No Response]"
            if pd.isnull(row.output) or row.output.strip() == "":
                df.at[row.Index, "output"] = "[No Response]"
            else:
                # If the "output" has "Conrad", replace it with 'person_name'
                if 'Conrad' in row.output:
                    df.at[row.Index, "output"] = row.output.replace("Conrad", person_name)
                # If the "output" has "Logos", replace it with 'model_name'
                if 'Logos' in row.output:
                    df.at[row.Index, "output"] = row.output.replace("Logos", model_name)

        # Convert the DataFrame values to a list of lists and return it
        return df.values.tolist()

    except Exception as e:  # Catch any exception during the loading process
        # Print an error message to the console
        print(f"Failed to load Course {path}: {str(e)}")
        return None  # Return None to indicate loading failure
