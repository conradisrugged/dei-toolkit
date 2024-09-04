import torch
from torch.utils.data import DataLoader
import random
import os
import re
import traceback
from queue import Queue
from .bootstrap import get_device
from .dataset import CustomDataset, load_dataset
from .evaluation import DEIEvaluation, evaluate_performance, adjust_weights


def clean_name(name):
    """Cleans a name string to make it safe for use as a filename."""
    name = re.sub(r'[/:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name


def educate(creation_window, model, tokenizer, epochs, inputs_per_miniepoch, datasets,
            dataset_weights, person_name, model_name, test_prompt, gui_update_queue,
            console_update_queue, master, device):
    """Guides the DEI's education (training) process."""

    # 1. Get the Appropriate Computing Device (CPU or GPU)
    device = get_device()

    # 2. Send a Message to the Console Indicating the Device Being Used
    console_update_queue.put((f"--- Device: {device} ---\n", "console", "Creation"))

    # 3. Initialize the DEIEvaluation Object
    evaluation = DEIEvaluation(model_name)

    # 4. Send Messages to the Console with Initial Training Parameters
    console_update_queue.put(
        (f"--- Initial Learning Rate: {evaluation.learning_rate:.4f} ---\n", "console", "Creation"))
    console_update_queue.put((f"--- Initial Batch Size: {evaluation.batch_size} ---\n", "console", "Creation"))

    # 5. Send a Message to the Console Indicating the Start of Education
    console_update_queue.put((f"--- EDUCATION OF DIGITAL EVOLVING INTELLIGENCE ---\n", "console", "Creation"))
    console_update_queue.put((f"--- Starting education for {epochs} Years ---\n", "console", "Creation"))

    # 6. Start the Education Process (Training Loop)
    try:
        # 6.1 Iterate over Each Epoch (Year)
        for epoch in range(epochs):
            # 6.2 Send a Message to the Console Indicating the Start of the Year
            console_update_queue.put((f"\n--- YEAR {epoch + 1} COMMENCED ---\n", "console", "Creation"))

            # 6.3 Calculate and Update the Year Progress Bar
            year_progress = int((epoch / epochs) * 100)
            gui_update_queue.put({"epoch": epoch + 1, "year_progress": year_progress})

            # 6.4 Iterate over Each Mini-Epoch (Term) within the Year
            for mini_epoch in range(4):
                # 6.4.1 Check if the Stop Training Event is Set
                if creation_window.stop_training_event.is_set():
                    # 6.4.1.1 If Stop Event is Set, Send a Message to the Console and Return
                    console_update_queue.put(("Training aborted.\n", "console", "Creation"))
                    return

                # 6.4.2 Send a Message to the Console Indicating the Start of the Term
                console_update_queue.put((f"\n--- TERM {mini_epoch + 1} ---\n", "console", "Creation"))

                # 6.4.3 Calculate and Update the Term Progress Bar
                term_progress = int(((mini_epoch + 1) / 4) * 100)
                gui_update_queue.put({"epoch": epoch + 1, "mini_epoch": mini_epoch, "term_progress": term_progress})

                # 6.4.4 Initialize an Empty List to Store Mini-Epoch Data
                mini_epoch_data = []

                # 6.4.5 Iterate over Each Dataset (Course)
                for i, dataset in enumerate(datasets):
                    # 6.4.5.1 Calculate the Number of Samples to Draw from the Dataset Based on its Weight
                    num_samples = max(2, int(inputs_per_miniepoch * dataset_weights[i]))

                    # 6.4.5.2 Send a Message to the Console Indicating the Number of Samples Being Drawn from the Course
                    console_update_queue.put((
                        f"--- Drawing {num_samples} Modules & Sub-Modules from Course {i + 1} ---\n",
                        "console",
                        "Creation"))

                    # 6.4.5.3 Add Randomly Sampled Data from the Dataset to the Mini-Epoch Data
                    mini_epoch_data.extend(random.sample(dataset, min(num_samples, len(dataset))))

                # 6.4.6 Send a Message to the Console Indicating the Total Number of Sub-Modules for the Term
                console_update_queue.put((
                    f"--- Total Sub-Modules for Term {mini_epoch + 1}: {len(mini_epoch_data)} ---\n",
                    "console",
                    "Creation"))

                # 6.4.7 Create a Custom PyTorch Dataset from the Mini-Epoch Data
                train_dataset = CustomDataset(mini_epoch_data, tokenizer)

                # 6.4.8 Create a PyTorch DataLoader from the Dataset
                train_dataloader = DataLoader(train_dataset, batch_size=evaluation.batch_size, shuffle=True)

                # 6.4.9 Set the Model to Training Mode
                model.train()

                # 6.4.10 Initialize Total Loss for the Term
                total_loss = 0

                # 6.4.11 Send a Message to the Console Indicating the Start of the Training Loop for the Term
                console_update_queue.put(
                    (f"--- Starting Training Loop for Term {mini_epoch + 1} ---\n", "console", "Creation"))

                # 6.4.12 Initialize Sub-Module Counter
                sub_module_counter = 0

                # 6.4.13 Get the Total Number of Sub-Modules in the DataLoader
                total_sub_modules = len(train_dataloader)

                # 6.4.14 Initialize the AdamW Optimizer
                optimizer = torch.optim.AdamW(model.parameters(), lr=evaluation.learning_rate)

                # 6.4.15 Iterate Over Each Batch of Data in the DataLoader
                for step, batch in enumerate(train_dataloader):
                    # 6.4.15.1 Check if the Stop Training Event is Set (if the user wants to abort training)
                    if creation_window.stop_training_event.is_set():
                        # 6.4.15.1.1 Send a "Training Aborted" Message to the Console and Return from the Function
                        console_update_queue.put(("Training aborted.\n", "console", "Creation"))
                        return

                    # 6.4.15.2 Break the Loop if the Number of Steps Reaches the Specified Limit (inputs_per_miniepoch)
                    if step >= inputs_per_miniepoch:
                        break

                    # 6.4.15.3 Send a Message to the Console Indicating that a New Sub-Module has Started
                    console_update_queue.put((f"       Sub-Module {step + 1} Started\n", "console", "Creation"))

                    # 6.4.15.4 Move Input Data to the Selected Device (CPU or GPU)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    # 6.4.15.5 Reset Gradients to Zero Before Backpropagation
                    optimizer.zero_grad()

                    # 6.4.15.6 Forward Pass: Get Model Outputs for the Current Batch
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                    # 6.4.15.7 Get the Loss Value from the Model Outputs
                    loss = outputs.loss

                    # 6.4.15.8 Print the Loss Value to the Console
                    console_update_queue.put(
                        (f"       Knowledge Gap Closure: {loss.item():.4f}\n", "console", "Creation"))

                    # 6.4.15.9 Add the Current Loss to the Total Loss for the Term
                    total_loss += loss.item()

                    # 6.4.15.10 Backpropagation: Calculate Gradients of the Loss with Respect to Model Parameters
                    loss.backward()

                    # 6.4.15.11 Update Model Parameters Using the Optimizer
                    optimizer.step()

                    # 6.4.15.12 Generate a Sample Output from the Model Using the Test Prompt
                    sample_output = evaluation.generate_sample_output(model, tokenizer, test_prompt)

                    # 6.4.15.13 Print a Message to the Console Indicating the Completion of the Sub-Module
                    console_update_queue.put(
                        (f"       Sub-Module {step + 1} Completed\n\n", "console", "Creation"))

                    # 6.4.15.14 Calculate the Progress of Sub-Modules for the Term
                    sub_module_progress = int(((step + 1) / total_sub_modules) * 100)

                    # 6.4.15.15 Print More Detailed Training Progress Every 10 Steps
                    if step % 10 == 0:
                        console_update_queue.put((
                            f"Year: {epoch + 1}\n"
                            f"       Term: {mini_epoch + 1}\n"
                            f"       Sub-Module: {step + 1}/{len(train_dataloader)}\n"
                            f"       Knowledge Gap Closure: {loss.item():.4f}\n"
                            f"       Learning Rate: {evaluation.learning_rate:.6f}\n"
                            f"       Batch Size: {evaluation.batch_size}\n\n",
                            "console", "Creation"
                        ))

                    # 6.4.15.16 Send GUI Updates to the Main Thread (using the gui_update_queue)
                    gui_update_queue.put({
                        "epoch": epoch + 1,
                        "mini_epoch": mini_epoch,
                        "sub_module_counter": sub_module_counter,
                        "total_sub_modules": total_sub_modules,
                        "sample_output": sample_output,
                        "term_progress": term_progress,
                        "year_progress": year_progress
                    })

                    # 6.4.15.17 Increment the Sub-Module Counter
                    sub_module_counter += 1

                # 6.4.16 Calculate the Average Loss for the Term
                avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0

                # 6.4.17 Send the Average Loss Value to the Console
                console_update_queue.put((
                    f"\nAverage Knowledge Gap Closure for Term {mini_epoch + 1}: {avg_loss:.4f}\n",
                    "console",
                    "Creation"))

                # 6.4.18 Update the DEI's Knowledge Radius based on a Random Value (between 0.8 and 1.0)
                evaluation.update_knowledge(random.uniform(0.8, 1.0))

                # 6.4.19 Evaluate the Model's Performance on the Datasets Used in This Mini-Epoch
                dataset_performance = evaluate_performance(model, tokenizer, datasets, mini_epoch_data,
                                                         model_name)

                # 6.4.20 Extract the Perplexity Score from the Evaluation Results
                perplexity_score = dataset_performance.get('perplexity_scores', [0])[0]

                # 6.4.21 Update the DEI's Understanding Depth based on the Perplexity Score
                evaluation.update_understanding(perplexity_score)

                # 6.4.22 Adjust the Weights of the Datasets based on Their Performance (Perplexity)
                dataset_weights = adjust_weights(dataset_weights, dataset_performance)

                # 6.4.23 Print the Adjusted Weights of Each Dataset (Course) to the Console
                for i, weight in enumerate(dataset_weights):
                    console_update_queue.put((f"Course {i + 1} Weight: {weight:.2f}\n", "console", "Creation"))

                # 6.4.24 Update the DEI's Learning Parameters (Learning Rate and Batch Size) based on Performance
                evaluation.update_learning_parameters(avg_loss, perplexity_score)

                # 6.4.25 Create a Dictionary to Store Data for the Term Card
                term_card_data = {
                    "epoch": epoch + 1,  # The current epoch (year)
                    "mini_epoch": mini_epoch,  # The current mini-epoch (term)
                    "knowledge_radius": evaluation.knowledge_radius,  # The DEI's current knowledge radius
                    "understanding_depth": evaluation.understanding_depth,
                    # The DEI's current understanding depth
                    "improvement_momentum": evaluation.improvement_momentum,
                    # The DEI's current improvement momentum
                    "report": evaluation.generate_report(epoch + 1, mini_epoch),
                    # Generate a textual report of the DEI's progress
                    "sample_output": sample_output  # A sample output generated by the DEI using the test prompt
                }

                # 6.4.26 Update epoch_data with lock (atomically)
                with creation_window.term_frames_lock:
                    if epoch not in creation_window.epoch_data:
                        creation_window.epoch_data[epoch] = {}
                    creation_window.epoch_data[epoch][mini_epoch] = term_card_data

                    # Send "MINI_EPOCH_COMPLETE" signal (new signal type)
                    gui_update_queue.put({"MINI_EPOCH_COMPLETE": epoch + 1, "mini_epoch": mini_epoch})

            # 6.5 Call next_epoch AFTER the mini-epoch loop (at the end of the epoch)
            creation_window.next_epoch()

            # 6.6 Send a Message to the Console Indicating the End of the Current Year (Epoch)
            console_update_queue.put((f"\n--- YEAR {epoch + 1} CONCLUDED ---\n", "console", "Creation"))

            # 6.7 Check for Training Abortion
            if creation_window.stop_training_event.is_set():
                console_update_queue.put(("Training aborted.\n", "console", "Creation"))
                return

        # 7. Send a Message to the Console Indicating that the Entire Education Process is Completed
        console_update_queue.put(("--- Education completed ---\n", "console", "Creation"))

        # 8. Save the Trained Model and Tokenizer
        try:
            output_model_path = os.path.join(creation_window.output_dir.get(),
                                             model_name)  # Construct the path to save the model
            os.makedirs(output_model_path, exist_ok=True)  # Create the directory if it doesn't exist
            model.save_pretrained(output_model_path)  # Save the trained model
            tokenizer.save_pretrained(output_model_path)  # Save the tokenizer
            console_update_queue.put((f"Model saved to: {output_model_path}\n", "console",
                                       "Creation"))  # Send a confirmation message to the console
        except Exception as e:
            # Handle any exceptions that might occur during model saving
            console_update_queue.put((f"Error saving model: {e}\n", "console", "Creation"))

    # 9. Handle Exceptions During the Entire Education Process
    except Exception as e:
        console_update_queue.put((f"Error during training: {e}\n{traceback.format_exc()}", "console", "Creation"))
        gui_update_queue.put(
            (lambda: messagebox.showerror("Error", f"Error during training: {e}"), "console", "Creation"))
