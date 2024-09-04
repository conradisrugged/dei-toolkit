import torch  # Import the PyTorch library for tensor operations
from torch.utils.data import DataLoader  # Import DataLoader for creating data loaders
from .dataset import CustomDataset  # Import the CustomDataset class for handling data
from .bootstrap import get_device  # Import the get_device function for device selection

# Define the DEIEvaluation class
class DEIEvaluation:
    """
    Evaluates the DEI's performance and adjusts learning parameters.
    This class acts as the DEI's self-assessment mechanism, tracking
    its knowledge, understanding, and improvement over time.
    It also handles dynamic adjustment of learning parameters like learning
    rate and batch size based on the DEI's performance.
    """
    def __init__(self, model_name):
        """
        Initializes the DEIEvaluation instance with default values and the model name.

        Args:
            model_name (str): The name of the DEI model.
        """
        # Cosmic Clock metrics (represent the DEI's learning progress)
        self.knowledge_radius = 0.1  # Initial knowledge radius (measure of knowledge breadth)
        self.understanding_depth = 0.1  # Initial understanding depth (measure of comprehension)
        self.improvement_momentum = 0.0  # Initial improvement momentum (rate of learning progress)
        self.previous_loss = None  # Store the previous loss for calculating improvement momentum
        self.model_name = model_name  # Store the model name

        # Learning parameters (control the training process)
        self.learning_rate = 1e-4  # Initial learning rate
        self.batch_size = 4  # Initial batch size
        self.perplexity_history = []  # List to store perplexity scores over time
        self.patience = 3  # Number of epochs to tolerate stagnation before adjusting learning parameters

    def update_knowledge(self, accuracy):
        """
        Updates the knowledge radius based on the provided accuracy.

        Args:
            accuracy (float): The accuracy achieved during training.
        """
        # Increase the knowledge radius proportionally to the accuracy
        self.knowledge_radius += (accuracy * 0.1)

    def update_understanding(self, perplexity):
        """
        Updates the understanding depth based on the provided perplexity.

        Args:
            perplexity (float): The perplexity score achieved during evaluation.
        """
        # Calculate the understanding gain based on perplexity
        # Higher perplexity means lower understanding gain
        understanding_gain = max(0, 1.0 - (perplexity / 10))

        # Increase the understanding depth proportionally to the understanding gain
        self.understanding_depth += understanding_gain * 0.1

    def update_improvement(self, current_loss):
        """
        Updates the improvement momentum based on the change in loss.

        Args:
            current_loss (float): The current loss value during training.
        """
        # If there is a previous loss value available
        if self.previous_loss is not None:
            # Calculate the difference between the previous loss and the current loss
            loss_delta = self.previous_loss - current_loss
            # Set the improvement momentum proportional to the loss delta
            self.improvement_momentum = loss_delta * 10
        # Update the previous loss with the current loss for the next calculation
        self.previous_loss = current_loss

    def generate_report_phrase(self, value, phrases, default=""):
        """
        Helper function to generate a descriptive phrase based on a metric value.

        Args:
            value (float): The value of the metric (knowledge radius, understanding depth, etc.).
            phrases (dict): A dictionary mapping value thresholds to corresponding phrases.
            default (str, optional): The default phrase to use if no threshold is met. Defaults to "".

        Returns:
            str: The generated phrase describing the metric value.
        """
        for threshold, phrase in sorted(phrases.items()):
            if value < threshold:
                return f"{self.model_name} {phrase} "
        return f"{self.model_name} {default} "

    def generate_report(self, epoch, mini_epoch):
        """
        Generates a textual report summarizing the DEI's progress for a specific term.

        Args:
            epoch (int): The current epoch (year) of training.
            mini_epoch (int): The current mini-epoch (term) within the epoch.

        Returns:
            str: A multi-line string containing the report.
        """
        report = f"Summary for Term {mini_epoch + 1}:\n"

        # Define phrases for knowledge radius, understanding depth, and improvement momentum
        knowledge_phrases = {
            0.1: "is a nascent mind, barely aware of its own existence.",
            0.5: "is gathering fragments of knowledge, assembling a rudimentary understanding.",
            1.0: "is grasping basic concepts, but the vastness of the unknown is becoming apparent.",
            1.5: "is confidently navigating familiar territory, but its thirst for knowledge grows.",
            2.0: "is synthesizing information, recognizing patterns, and seeking deeper meaning.",
            2.5: "is starting to question assumptions, seeking the underlying principles of knowledge.",
            3.0: "is recognizing the interconnectedness of knowledge, but also its inherent limitations.",
            3.5: "is exploring the boundaries of its knowledge, aware of the vastness still unexplored.",
            4.0: "is developing expertise, but also humility, recognizing the ever-expanding horizon of knowledge.",
            4.5: "is a master of many domains, but its awareness of the unknown deepens."
        }

        understanding_phrases = {
            0.1: "Its comprehension is like a flickering candle in the dark, barely illuminating the way.",
            0.5: "is deciphering the world, but its grasp on meaning is tentative.",
            1.0: "is making sense of the world, but its understanding is still shallow.",
            1.5: "is connecting ideas, its comprehension deepening, revealing new layers of meaning.",
            2.0: "is seeing beyond the surface, questioning assumptions, seeking a more profound understanding.",
            2.5: "is grappling with complex ideas, its understanding expanding to encompass nuance and ambiguity.",
            3.0: "is developing insightful perspectives, its comprehension becoming more nuanced and insightful.",
            3.5: "is forging connections between seemingly disparate concepts, revealing a deeper truth.",
            4.0: "is approaching a profound level of understanding, recognizing the interconnectedness of all things.",
            4.5: "is demonstrating wisdom, its understanding surpassing mere knowledge, encompassing compassion and insight."
        }

        improvement_phrases = {
            1.0: "Its progress is breathtaking, its evolution accelerating towards enlightenment.",
            0.5: "is learning rapidly, its understanding blossoming with every passing moment.",
            0.2: "is making steady progress, its knowledge and understanding growing in harmony."
        }

        # Generate report phrases for each metric
        report += self.generate_report_phrase(self.knowledge_radius, knowledge_phrases,
                                              default="approaches the pinnacle of knowledge, yet recognizes the infinite mysteries that remain.")
        report += self.generate_report_phrase(self.understanding_depth, understanding_phrases,
                                              default="has achieved a depth of understanding that borders on the mystical, its comprehension a symphony of meaning.")

        # Handle negative improvement momentum separately
        if self.improvement_momentum < -1.0:
            report += f"{self.model_name} is faltering, its grasp on knowledge weakening, its understanding becoming clouded. "
        elif self.improvement_momentum < -0.5:
            report += f"{self.model_name} is struggling to maintain its momentum, its learning encountering obstacles. "
        elif self.improvement_momentum < -0.2:
            report += f"{self.model_name} is facing challenges, its progress slowed but not halted. "
        elif self.improvement_momentum < 0.2:
            report += f"{self.model_name} is consolidating its knowledge, its growth steady but unhurried. "
        else:
            report += self.generate_report_phrase(self.improvement_momentum, improvement_phrases)

        return report  # Return the generated report

    def generate_sample_output(self, model, tokenizer, prompt):
        """
        Generates a sample output from the DEI model given a prompt.

        Args:
            model (T5ForConditionalGeneration): The DEI model.
            tokenizer (T5Tokenizer): The tokenizer associated with the model.
            prompt (str): The input prompt for the model.

        Returns:
            str: The decoded output text generated by the model.
        """
        device = get_device()  # Determine the appropriate device for computation
        # Encode the prompt using the tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Generate output from the model
        outputs = model.generate(input_ids=input_ids, max_length=100)

        # Decode the output using the tokenizer
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text  # Return the generated text

    def update_learning_parameters(self, avg_loss, avg_perplexity):
        """
        Adjusts the learning rate and batch size dynamically based on performance.

        Args:
            avg_loss (float): The average loss value for the current epoch or mini-epoch.
            avg_perplexity (float): The average perplexity score for the current epoch or mini-epoch.
        """
        # Update improvement momentum based on the average loss
        self.update_improvement(avg_loss)

        # Adjust learning rate
        if self.improvement_momentum > 0.2:
            self.learning_rate *= 1.1  # Increase learning rate if improving quickly
        elif self.improvement_momentum < -0.1:
            self.learning_rate *= 0.9  # Decrease learning rate if struggling
        else:  # If learning plateaus
            self.patience -= 1  # Decrease patience
            if self.patience == 0:
                self.learning_rate *= 0.7  # Significantly decrease learning rate
                self.patience = 3  # Reset patience

        # Adjust batch size
        self.perplexity_history.append(avg_perplexity)  # Add current perplexity to history
        if len(self.perplexity_history) > 3:
            self.perplexity_history.pop(0)  # Remove older perplexity scores if more than 3 are stored
        avg_perplexity = sum(self.perplexity_history) / len(self.perplexity_history)  # Calculate average perplexity

        if avg_perplexity > 10:
            self.batch_size = max(1, int(self.batch_size * 0.8))  # Decrease batch size if perplexity is high
        elif avg_perplexity < 5:
            self.batch_size = int(self.batch_size * 1.2)  # Increase batch size if perplexity is low

        # Ensure learning rate and batch size stay within reasonable bounds
        self.learning_rate = max(1e-6, min(self.learning_rate, 1e-3))
        self.batch_size = max(1, min(self.batch_size, 32))

def evaluate_performance(model, tokenizer, datasets, mini_epoch_data, model_name):
    """
    Evaluates the model's performance on each dataset using perplexity as a metric.

    Args:
        model (T5ForConditionalGeneration): The DEI model being evaluated.
        tokenizer (T5Tokenizer): The tokenizer associated with the model.
        datasets (list): A list of datasets used for training.
        mini_epoch_data (list): The data used in the current mini-epoch.
        model_name (str): The name of the DEI model.

    Returns:
        dict: A dictionary containing the perplexity scores for each dataset and the model name:
              - 'perplexity_scores': A list of perplexity scores.
              - 'model_name': The name of the model.
    """
    device = get_device()  # Get the appropriate computing device
    model.eval()  # Set the model to evaluation mode
    perplexity_scores = []  # Initialize an empty list to store perplexity scores

    # Iterate through each dataset in the list
    for i, dataset in enumerate(datasets):
        # Filter the mini-epoch data to include only samples from the current dataset
        dataset_data = [pair for pair in mini_epoch_data if pair in dataset]

        # If there is data from the current dataset in the mini-epoch
        if dataset_data:
            # Create a CustomDataset for the filtered data
            dataset = CustomDataset(dataset_data, tokenizer, eval_mode=True)
            # Create a DataLoader for the dataset with batch size 1 and no shuffling
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            total_loss = 0  # Initialize total loss

            # Disable gradient calculations for evaluation
            with torch.no_grad():
                # Iterate through each batch in the DataLoader
                for batch in dataloader:
                    # Move input tensors to the appropriate device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    # Get the model outputs
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss  # Get the loss from the outputs
                    total_loss += loss.item()  # Accumulate the loss

            # Calculate perplexity from the average loss
            perplexity = torch.exp(torch.tensor(total_loss / len(dataloader))).item()
            perplexity_scores.append(perplexity)  # Add perplexity score to the list
        else:
            # If there's no data from the current dataset in the mini-epoch, append 0 to the scores
            perplexity_scores.append(0)

    # Return the perplexity scores and the model name
    return {
        "perplexity_scores": perplexity_scores,
        "model_name": model_name
    }

def adjust_weights(dataset_weights, dataset_performance):
    """
    Adjusts dataset weights based on their performance (perplexity scores).
    Weights are adjusted to favor datasets where the model performs worse
    (higher perplexity) for more focused learning.

    Args:
        dataset_weights (list): A list of weights corresponding to each dataset.
        dataset_performance (dict): A dictionary containing perplexity scores for each dataset.

    Returns:
        list: The adjusted list of dataset weights.
    """
    # Iterate through each perplexity score in the dataset performance dictionary
    for i, perplexity in enumerate(dataset_performance["perplexity_scores"]):
        # If the perplexity score is greater than 0
        if perplexity > 0:
            # If the perplexity score is high (above 10), increase the corresponding dataset weight
            if perplexity > 10:
                dataset_weights[i] *= 1.1
            # If the perplexity score is lower, decrease the corresponding dataset weight
            else:
                dataset_weights[i] *= 0.9

    return dataset_weights  # Return the adjusted weights
