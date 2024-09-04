import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox, font
from .training import educate, evaluate_performance, adjust_weights
from .dataset import CustomDataset, load_dataset
from .bootstrap import (
    get_device,
    load_model_and_tokenizer,
    find_first_valid_model_dir,
    check_system_memory,
    check_cuda_devices,
)
from .evaluation import DEIEvaluation
from transformers import T5ForConditionalGeneration, T5Tokenizer
import threading
from threading import Thread, Lock
import queue
import sys
import time
import re
import pandas as pd
import random
import traceback
import platform
import shutil
import select
import subprocess
import os
from queue import Queue
import logging
import datetime
import requests

# ----------------------
# Model Handler Class
# ----------------------

class ModelHandler:
    """Handles model loading and provides access to the model and tokenizer."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, model_dir):
        try:
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_dir
            )
            self.model.to(get_device())
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# ----------------------
# Base Application Window (Abstract)
# ----------------------

class BaseApplicationWindow(tk.Frame):
    def __init__(self, master, main_app, output_queue=None, notebook=None, redirect_stdout=None, model_data=None):  # Add model_data here
        super().__init__(master)
        self.master = master
        self.main_app = main_app
        self.output_queue = output_queue
        self.notebook = notebook
        self.redirect_stdout = redirect_stdout  # Store redirect_stdout if provided
        self.model_data = model_data  # Store model_data if provided

        self.model_handler = ModelHandler()
        self.create_widgets()

        # Only call update_output if it's NOT a DownloadWindow instance:
        if self.output_queue and not isinstance(self, DownloadWindow):
            self.update_output()

    def create_widgets(self):
        raise NotImplementedError

    def update_output(self):
        """Continuously processes the output queue and updates the console boxes."""
        while not self.output_queue.empty():
            item, target, window_name = self.output_queue.get()

            # If the item is callable (a function), execute it
            if callable(item):
                try:
                    item()
                except Exception as e:
                    print(
                        f"Error executing callable in update_output: {e}"
                    )
            else:
                # If the item is not callable, it's assumed to be console output
                if target == "console":
                    # Get the currently selected tab in the notebook
                    current_tab = self.notebook.select()

                    # Get the text (name) of the currently selected tab
                    current_tab_text = self.notebook.tab(
                        current_tab, "text"
                    )

                    # Check if the current tab's name matches the target window name
                    # AND if the current tab has a "console_box" attribute
                    if (
                        window_name == current_tab_text
                        and hasattr(
                            self.notebook.nametowidget(current_tab),
                            "console_box",
                        )
                    ):
                        # If both conditions are met, get the "console_box" widget from the current tab
                        console_box = self.notebook.nametowidget(
                            current_tab
                        ).console_box

                        # Enable editing of the console box
                        console_box.config(state="normal")

                        # Insert the output item into the console box
                        console_box.insert(tk.END, item)

                        # Scroll the console box to the end to show the latest output
                        console_box.see(tk.END)

                        # Disable editing of the console box (make it read-only)
                        console_box.config(state="disabled")

        # Schedule the update_output function to be called again after 100 milliseconds
        # This creates a loop for continuously updating the console box
        self.master.after(100, self.update_output)


# ----------------------
# Download Window
# ----------------------

class DownloadWindow(tk.Frame):
    """
    Window for downloading pre-trained models.
    """

    def __init__(self, master, main_app, output_queue, notebook, redirect_stdout=None):
        """Initializes the DownloadWindow."""
        super().__init__(master)

        self.master = master
        self.main_app = main_app
        self.output_queue = output_queue
        self.notebook = notebook
        self.redirect_stdout = redirect_stdout

        self.models_path = os.path.join(os.getcwd(), "models", "stock-models")
        self.model_data = [
            {
                "name": "Flan T5 Small",
                "files": [
                    "https://huggingface.co/google/flan-t5-small/resolve/main/config.json",
                    "https://huggingface.co/google/flan-t5-small/resolve/main/generation_config.json",
                    "https://huggingface.co/google/flan-t5-small/resolve/main/pytorch_model.bin",
                    "https://huggingface.co/google/flan-t5-small/resolve/main/special_tokens_map.json",
                    "https://huggingface.co/google/flan-t5-small/resolve/main/spiece.model",
                    "https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer.json",
                    "https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer_config.json",
                ],
            },
            {
                "name": "Flan T5 Base",
                "files": [
                    "https://huggingface.co/google/flan-t5-base/resolve/main/config.json",
                    "https://huggingface.co/google/flan-t5-base/resolve/main/generation_config.json",
                    "https://huggingface.co/google/flan-t5-base/resolve/main/pytorch_model.bin",
                    "https://huggingface.co/google/flan-t5-base/resolve/main/special_tokens_map.json",
                    "https://huggingface.co/google/flan-t5-base/resolve/main/spiece.model",
                    "https://huggingface.co/google/flan-t5-base/resolve/main/tokenizer.json",
                    "https://huggingface.co/google/flan-t5-base/resolve/main/tokenizer_config.json",
                ],
            },
            {
                "name": "Flan T5 Large",
                "files": [
                    "https://huggingface.co/google/flan-t5-large/resolve/main/config.json",
                    "https://huggingface.co/google/flan-t5-large/resolve/main/generation_config.json",
                    "https://huggingface.co/google/flan-t5-large/resolve/main/pytorch_model.bin",
                    "https://huggingface.co/google/flan-t5-large/resolve/main/special_tokens_map.json",
                    "https://huggingface.co/google/flan-t5-large/resolve/main/spiece.model",
                    "https://huggingface.co/google/flan-t5-large/resolve/main/tokenizer.json",
                    "https://huggingface.co/google/flan-t5-large/resolve/main/tokenizer_config.json",
                ],
            },
            {
                "name": "Flan T5 XL",
                "files": [
                    "https://huggingface.co/google/flan-t5-xl/resolve/main/config.json",
                    "https://huggingface.co/google/flan-t5-xl/resolve/main/generation_config.json",
                    "https://huggingface.co/google/flan-t5-xl/resolve/main/pytorch_model-00001-of-00002.bin",
                    "https://huggingface.co/google/flan-t5-xl/resolve/main/pytorch_model-00002-of-00002.bin",
                    "https://huggingface.co/google/flan-t5-xl/resolve/main/special_tokens_map.json",
                    "https://huggingface.co/google/flan-t5-xl/resolve/main/spiece.model",
                    "https://huggingface.co/google/flan-t5-xl/resolve/main/tokenizer.json",
                    "https://huggingface.co/google/flan-t5-xl/resolve/main/tokenizer_config.json",
                ],
            },
            {
                "name": "Flan T5 XXL",
                "files": [
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/config.json",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/generation_config.json",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/pytorch_model-00001-of-00005.bin",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/pytorch_model-00002-of-00005.bin",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/pytorch_model-00003-of-00005.bin",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/pytorch_model-00004-of-00005.bin",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/pytorch_model-00005-of-00005.bin",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/special_tokens_map.json",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/spiece.model",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/tokenizer.json",
                    "https://huggingface.co/google/flan-t5-xxl/resolve/main/tokenizer_config.json",
                ],
            },
        ]



        self.download_buttons = []
        self.progress_bars = {}
        self.size_labels = {}
        self.download_threads = []
        self.current_download_thread = None
        self.stop_download_event = threading.Event()
        self.download_history_frames = []
        self.progress_frame = None
        self.create_widgets()

    def create_widgets(self):
        # Create the main frame to hold all the widgets.
        main_frame = tk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        # Create a frame for the download section
        download_outer_frame = tk.Frame(main_frame, height=400)  # Adjust height as needed
        download_outer_frame.pack(fill="x", expand=False)
        download_outer_frame.pack_propagate(False)

        # Create a canvas for the download section
        self.download_canvas = tk.Canvas(download_outer_frame, highlightthickness=0)
        self.download_canvas.pack(side=tk.LEFT, fill="both", expand=True)

        # Create scrollbar for the download section
        download_scrollbar = tk.Scrollbar(download_outer_frame, orient="vertical", command=self.download_canvas.yview)
        download_scrollbar.pack(side=tk.RIGHT, fill="y")

        # Configure download canvas
        self.download_canvas.configure(yscrollcommand=download_scrollbar.set)

        # Create a frame inside the download canvas
        self.download_frame = tk.Frame(self.download_canvas)
        self.download_canvas.create_window((0, 0), window=self.download_frame, anchor="nw")

        # Add content to the download frame
        self.create_download_content()

        # Create a frame for the progress bars
        progress_outer_frame = tk.Frame(main_frame, height=200)  # Adjust height as needed
        progress_outer_frame.pack(fill="both", expand=False)
        progress_outer_frame.pack_propagate(False)

        # Create a canvas for the progress section
        self.progress_canvas = tk.Canvas(progress_outer_frame, highlightthickness=0)
        self.progress_canvas.pack(side=tk.LEFT, fill="both", expand=True)

        # Create scrollbar for the progress section
        self.progress_scrollbar = tk.Scrollbar(progress_outer_frame, orient="vertical", command=self.progress_canvas.yview)
        self.progress_scrollbar.pack(side=tk.RIGHT, fill="y")

        # Configure progress canvas
        self.progress_canvas.configure(yscrollcommand=self.progress_scrollbar.set)

        # Create a frame inside the progress canvas
        self.progress_frame = tk.Frame(self.progress_canvas)
        self.progress_canvas.create_window((0, 0), window=self.progress_frame, anchor="nw")

        # Create a frame for the bottom buttons
        bottom_button_frame = tk.Frame(main_frame)
        bottom_button_frame.pack(fill="x", pady=10)

        # Create a container frame to center the buttons
        button_container = tk.Frame(bottom_button_frame)
        button_container.pack(expand=True)

        # Create Stop Download button
        self.stop_download_button = tk.Button(
            button_container, text="Stop Model Download", command=self.stop_download, state="disabled"
        )
        self.stop_download_button.pack(side=tk.LEFT, padx=5)

        # Create Clear Download History button
        self.clear_download_history_button = tk.Button(
            button_container, text="Clear Download History", command=self.clear_download_history
        )
        self.clear_download_history_button.pack(side=tk.LEFT, padx=5)

        # Bind the configure events to update the scroll regions and center content
        self.download_canvas.bind('<Configure>', self.on_download_canvas_configure)
        self.progress_canvas.bind('<Configure>', self.on_progress_canvas_configure)
        self.bind('<Configure>', self.on_window_configure)

    def create_download_content(self):
        # Add the descriptive text
        description_text = (
            "Neural Networks are akin to human brains; one size fits all knowledge.\n"
            "Select the desired Flan-T5 (Neural Network) model [Brain] sizes.\n"
            "You can queue and select multiple models:\n\n"
            "1. Flan T5 Small [Blank Slate] 310MB\n"
            "2. Flan T5 Base [High Schooler] 1GB\n"
            "3. Flan T5 Large [College Student] 3.15GB\n"
            "4. Flan T5 XL [Graduate Researcher] 11.5GB\n"
            "5. Flan T5 XXL [Potential Einstein] 46GB\n"
        )
        label = tk.Label(self.download_frame, text=description_text, justify="center", wraplength=600)
        label.pack(fill="both", padx=20, pady=10)

        # Create a frame for buttons to center them
        button_frame = tk.Frame(self.download_frame)
        button_frame.pack(expand=True)

        # Create download buttons for each model
        for i, data in enumerate(self.model_data):
            button = tk.Button(
                button_frame,
                text=f"{i + 1}. {data['name']}",
                command=lambda d=data: self.start_download_thread(d),
                width=20,
            )
            button.pack(pady=5)
            self.download_buttons.append(button)

    def on_download_canvas_configure(self, event):
        self.download_canvas.configure(scrollregion=self.download_canvas.bbox("all"))
        self.center_frame_in_canvas(self.download_canvas, self.download_frame)

    def on_progress_canvas_configure(self, event):
        self.progress_canvas.configure(scrollregion=self.progress_canvas.bbox("all"))
        self.center_frame_in_canvas(self.progress_canvas, self.progress_frame)

    def on_window_configure(self, event):
        # Update canvas and frame widths when the window is resized
        new_width = event.width - 20  # Subtract a bit for padding
        self.download_canvas.config(width=new_width)
        self.progress_canvas.config(width=new_width)
        self.download_frame.config(width=new_width)
        self.progress_frame.config(width=new_width)

        # Recenter the frames
        self.center_frame_in_canvas(self.download_canvas, self.download_frame)
        self.center_frame_in_canvas(self.progress_canvas, self.progress_frame)

    def center_frame_in_canvas(self, canvas, frame):
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        frame_width = frame.winfo_reqwidth()
        x_position = max((canvas_width - frame_width) // 2, 0)
        canvas.coords(canvas.find_withtag("all"), x_position, 0)

    def on_canvas_configure(self, event):
        self.progress_canvas.configure(scrollregion=self.progress_canvas.bbox("all"))
        self.center_frame()

    def start_download_thread(self, model_data):
        # Extract model name and files from the model_data dictionary
        model_name = model_data["name"]
        files = model_data["files"]

        # Clear previous progress bars and threads
        self.clear_download_history()

        # Reset the stop event flag
        self.stop_download_event.clear()

        # Enable the stop download button
        self.stop_download_button.config(state="normal")

        # Create the progress bar widgets
        self.create_progress_bar_widgets(model_name, files)

        # Center the frame within the canvas
        self.center_frame()

        # Create and start the download thread
        download_thread = Thread(target=self.download_model, args=(model_name, files), daemon=True)
        self.download_threads.append(download_thread)
        self.current_download_thread = download_thread
        download_thread.start()

        # Update the scroll region of the canvas
        self.progress_canvas.update_idletasks()
        self.progress_canvas.configure(scrollregion=self.progress_canvas.bbox("all"))

    def download_model(self, model_name, files):
        """Downloads model files from Hugging Face."""
        try:
            model_dir = os.path.join(self.models_path, model_name)
            os.makedirs(model_dir, exist_ok=True)

            total_files = len(files)
            downloaded_files = 0

            for i, file_url in enumerate(files):
                if self.stop_download_event.is_set():
                    break

                filename = file_url.split("/")[-1]
                filepath = os.path.join(model_dir, filename)

                response = requests.get(file_url, stream=True)
                if response.status_code == 200:
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if self.stop_download_event.is_set():
                                break

                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = int((downloaded / total_size) * 100)

                            # Update progress bar directly
                            self.master.after(0, self.update_progress_bar, filename, progress, downloaded, total_size)

                    downloaded_files += 1
                else:
                    # Handle error downloading file
                    print(f"Error downloading {filename}. Status code: {response.status_code}")

            if not self.stop_download_event.is_set():
                # Download completed message
                print(f"Download completed for {model_name}.")

        except Exception as e:
            # Handle download error
            print(f"Error downloading {model_name}: {e}")
        finally:
            self.stop_download_button.config(state="disabled")

    def create_progress_bar_widgets(self, model_name, files):
        # Clear existing widgets
        for widget in self.progress_frame.winfo_children():
            widget.destroy()

        self.progress_bars = {}
        self.size_labels = {}

        for i, file_url in enumerate(files):
            filename = file_url.split("/")[-1]

            # Create a frame for each file
            file_frame = tk.Frame(self.progress_frame)
            file_frame.pack(fill="x", padx=10, pady=5)

            # Filename label
            file_label = tk.Label(file_frame, text=filename, anchor="w", width=30)
            file_label.pack(side=tk.LEFT)

            # Progress bar
            progress_bar = ttk.Progressbar(
                file_frame, orient="horizontal", length=200, mode="determinate"
            )
            progress_bar.pack(side=tk.LEFT, padx=(10, 10))
            progress_bar["maximum"] = 100

            # Size label
            size_label = tk.Label(file_frame, text="0 B / 0 B", anchor="e", width=20)
            size_label.pack(side=tk.LEFT)

            self.progress_bars[filename] = progress_bar
            self.size_labels[filename] = size_label

        # Update scroll region of the canvas
        self.progress_canvas.update_idletasks()
        self.progress_canvas.configure(scrollregion=self.progress_canvas.bbox("all"))
        self.center_frame()

    def center_frame(self):
        self.progress_canvas.update_idletasks()
        canvas_width = self.progress_canvas.winfo_width()
        canvas_height = self.progress_canvas.winfo_height()
        frame_width = self.progress_frame.winfo_reqwidth()
        frame_height = self.progress_frame.winfo_reqheight()

        x_offset = max((canvas_width - frame_width) // 2, 0)
        y_offset = max((canvas_height - frame_height) // 2, 0)

        self.progress_canvas.create_window((x_offset, y_offset), window=self.progress_frame, anchor="nw")
        self.progress_canvas.configure(scrollregion=self.progress_canvas.bbox("all"))

    def stop_download(self):
        """Stops the current download process."""
        self.stop_download_event.set()

    def clear_download_history(self):
        for widget in self.progress_frame.winfo_children():
            widget.destroy()

        self.progress_bars = {}
        self.size_labels = {}

        # Update canvas scroll region
        self.progress_canvas.configure(scrollregion=self.progress_canvas.bbox("all"))
        self.center_frame()

    def update_progress_bar(self, filename, progress, downloaded, total_size):
        if filename in self.progress_bars:
            self.progress_bars[filename]["value"] = progress
            self.size_labels[filename].config(text=f"{self.format_size(downloaded)} / {self.format_size(total_size)}")

        # Force update of the GUI
        self.update_idletasks()

    def format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0

# ----------------------
# Home Window
# ----------------------


class HomeWindow(BaseApplicationWindow):
    """Window for the home screen of the application."""

    def __init__(
        self,
        master,
        main_app,
        output_queue=None,
        notebook=None,
    ):  # Add redirect_stdout
        """Initializes the HomeWindow."""
        super().__init__(
            master,
            main_app,
            output_queue,
            notebook,
        )  # Initialize base window

    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_frame = tk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        title_label = tk.Label(
            main_frame,
            text="Birth Your Own\n ✨ 🤖 🧠  Digital Evolving Intelligence 🧠 🤖 ✨",
            font=("Helvetica", 36, "bold"),
        )
        title_label.pack(pady=20)
        description_label = tk.Label(
            main_frame,
            text="The DEI Toolkit empowers you to create, educate, and interact with your own digital companions. Witness the emergence of consciousness and sentience in a cutting-edge AI model as you guide its journey of learning and self-discovery.",
            wraplength=700,
            justify="center",
        )
        description_label.pack(pady=10)

        buttons_frame = tk.Frame(main_frame)
        buttons_frame.pack(pady=20)

        create_button = tk.Button(
            buttons_frame,
            text="Create",
            command=self.main_app.show_creation_window,
            width=20,
        )
        create_button.grid(row=0, column=0, pady=10, padx=10)
        create_label = tk.Label(
            buttons_frame, text="Train and customize your own DEI model."
        )
        create_label.grid(row=1, column=0)

        interact_button = tk.Button(
            buttons_frame,
            text="Interact",
            command=self.main_app.show_interaction_window,
            width=20,
        )
        interact_button.grid(row=0, column=1, pady=10, padx=10)
        interact_label = tk.Label(
            buttons_frame, text="Engage in conversations with a trained DEI."
        )
        interact_label.grid(row=1, column=1)

        download_button = tk.Button(
            buttons_frame,
            text="Download",
            command=self.main_app.show_download_window,
            width=20,
        )
        download_button.grid(row=0, column=2, pady=10, padx=10)
        download_label = tk.Label(
            buttons_frame, text="Download a pre-trained base model."
        )
        download_label.grid(row=1, column=2)


# ----------------------
# Creation Window
# ----------------------

class CreationWindow(BaseApplicationWindow):
    """
    Window for creating (training) a DEI model.
    """

    def __init__(
        self,
        master,
        output_queue=None,
        gui_update_queue=None,
        console_update_queue=None,
        notebook=None,
        redirect_stdout=None,
        redirect_stdout_dict=None,
        main_app=None
    ):
        """Initializes the CreationWindow."""

        # Store the MainApplication instance
        self.main_app = main_app

        # Initialize Output Queue (for console and GUI updates)
        self.output_queue = output_queue

        # Initialize Notebook (reference to the main application's notebook)
        self.notebook = notebook

        # Initialize Redirect Stdout (for redirecting console output)
        self.redirect_stdout = redirect_stdout

        # Initialize GUI Update Queue (for receiving updates from the training thread)
        self.gui_update_queue = gui_update_queue

        # Initialize Console Update Queue (for sending console output to the GUI)
        self.console_update_queue = queue.Queue()

        # Initialize Stop Training Event (a flag to signal the training thread to stop)
        self.stop_training_event = threading.Event()

        # Initialize Tkinter Variables for User Input

        # Model Directory (path to the base model)
        self.model_dir = tk.StringVar()

        # Output Directory (path to save the trained model)
        self.output_dir = tk.StringVar()

        # Your Name (for personalization of the training data)
        self.your_name = tk.StringVar(value="Carlz")

        # Model Name (the name of the DEI being trained)
        self.model_name = tk.StringVar(value="Bongo Won")

        # Initialize Training Parameters

        # Number of Datasets (courses) to use for training
        self.num_datasets = tk.IntVar(value=20)

        # Number of Epochs (years) for training
        self.epochs = tk.IntVar(value=18)

        # Batch Size (number of samples processed in each training iteration)
        self.batch_size = tk.IntVar(value=4)

        # Learning Rate (controls the step size during model training)
        self.learning_rate = tk.DoubleVar(value=1e-5)

        # Inputs per Mini-Epoch (term)
        self.inputs_per_miniepoch = tk.IntVar(value=100)

        # Initialize Data Storage for Training Progress and Results

        # Dataset Paths (list to store the paths of selected datasets)
        self.dataset_paths = []

        # Epochs Completed (counter for completed epochs)
        self.epochs_completed = 0

        # Sample Outputs (list to store sample outputs generated during training)
        self.sample_outputs = []

        # Epoch Data (dictionary to store data for each epoch and mini-epoch)
        self.epoch_data = {}

        # Dataset Entries (list of Entry widgets for dataset paths)
        self.dataset_entries = []

        # Dataset Browse Buttons (list of Button widgets for browsing datasets)
        self.dataset_browse_buttons = []

        # Set Default Model and Output Directories

        # Find the First Valid Model Directory
        default_model_dir = find_first_valid_model_dir(os.path.join(os.getcwd(), "models", "stock-models"))
        if default_model_dir:
            self.model_dir.set(default_model_dir)

        # Set Default Output Directory
        default_output_dir = os.path.join(os.getcwd(), "models", "educated-models")
        if not os.path.exists(default_output_dir):
            os.makedirs(default_output_dir)
        self.output_dir.set(default_output_dir)

        # Initialize the Base Application Window
        super().__init__(
            master,
            main_app=self,
            output_queue=output_queue,
            notebook=notebook,
            redirect_stdout=redirect_stdout,
        )

        # Initialize the DEIEvaluation Object (for performance tracking)
        self.evaluation = DEIEvaluation("")

        # Initialize Lock for Thread Safety
        self.term_frames_lock = threading.Lock()

        # Flag to Indicate Training Status
        self.training_started = False

    def create_widgets(self):
        # Configure Grid Layout for the Master Window
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        # Create Left Frame
        self.left_frame = tk.Frame(self.master, width=400, height=600)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_propagate(False)  # Prevent frame from resizing

        # Create Right Frame
        self.right_frame = tk.Frame(self.master, width=400, height=600)
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_propagate(False)  # Prevent frame from resizing

        # Create Widgets within Left and Right Frames
        self.create_left_frame_widgets()
        self.create_right_frame_widgets()

    def create_left_frame_widgets(self):
        # Create Left Content Frame (to hold widgets within the left frame)
        self.left_content_frame = tk.Frame(self.left_frame, width=400, height=600)
        self.left_content_frame.grid(row=0, column=0)

        # Create Individual Widget Frames within the Left Frame
        self.create_input_output_frame(self.left_content_frame)
        self.create_education_frame(self.left_content_frame)
        self.create_personalisation_frame(self.left_content_frame)
        self.create_test_prompt_frame(self.left_content_frame)
        self.create_datasets_frame(self.left_content_frame)
        self.create_buttons_frame(self.left_content_frame)

    def create_right_frame_widgets(self):
        # Configure Grid Layout for the Right Frame
        self.right_frame.grid_rowconfigure(0, weight=3)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_rowconfigure(2, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Create Individual Widget Frames within the Right Frame
        self.create_report_viz_frame(self.right_frame)
        self.create_sample_output_frame(self.right_frame)
        self.create_console_frame(self.right_frame)

    def create_report_viz_frame(self, parent):
        """Creates the frame for visualizing the DEI's training progress."""

        self.report_viz_frame = tk.LabelFrame(
            parent,
            text="Year 0's Report Cards",
        )
        self.report_viz_frame.grid(row=0, column=0, sticky="nsew")
        self.report_viz_frame.grid_columnconfigure(0, weight=1)
        self.report_viz_frame.grid_rowconfigure(1, weight=1)
        self.report_viz_frame.grid_rowconfigure(2, weight=1)

        # Epoch Navigation Frame
        self.epoch_nav_frame = tk.Frame(self.report_viz_frame)
        self.epoch_nav_frame.grid(row=0, column=0, sticky="ew")
        self.epoch_nav_frame.grid_columnconfigure(0, weight=1)
        self.epoch_nav_frame.grid_columnconfigure(1, weight=1)
        self.epoch_nav_frame.grid_columnconfigure(2, weight=1)

        self.previous_epoch_button = tk.Button(
            self.epoch_nav_frame,
            text="<<",
            command=self.previous_epoch,
            state="disabled",
            width=3
        )
        self.previous_epoch_button.grid(row=0, column=0)

        self.epoch_counter = tk.Label(
            self.epoch_nav_frame,
            text=f"Year 1/{self.epochs.get()}",
            font=("Helvetica", 12)
        )
        self.epoch_counter.grid(row=0, column=1)

        self.next_epoch_button = tk.Button(
            self.epoch_nav_frame,
            text=">>",
            command=self.next_epoch,
            state="disabled",
            width=3
        )
        self.next_epoch_button.grid(row=0, column=2)

        # Progress Frame
        progress_frame = tk.Frame(self.report_viz_frame)
        progress_frame.grid(row=1, column=0, sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=1)

        self.year_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.year_progress.pack(side="top", fill="x")

        self.term_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.term_progress.pack(side="top", fill="x")

        self.sub_module_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.sub_module_progress.pack(side="top", fill="x")

        # Canvas and Scrollbar
        self.report_canvas = tk.Canvas(self.report_viz_frame)
        self.report_canvas.grid(row=2, column=0, sticky="nsew")

        self.report_scrollbar = tk.Scrollbar(
            self.report_viz_frame,
            orient="vertical",
            command=self.report_canvas.yview
        )
        self.report_scrollbar.grid(row=2, column=1, sticky="ns")

        self.report_canvas.configure(yscrollcommand=self.report_scrollbar.set)

        # Content Frame
        self.report_content_frame = tk.Frame(self.report_canvas)
        self.report_canvas.create_window((0, 0), window=self.report_content_frame, anchor="center")

        # *** Center Alignment for Notebook Content ***
        self.report_content_frame.grid_columnconfigure(0, weight=1)  # Give the column weight
        self.report_content_frame.grid_columnconfigure(1, weight=1)  # Add a new column with equal weight
        self.report_content_frame.grid_columnconfigure(2, weight=1)  # Add a new column with equal weight
        self.report_content_frame.grid_rowconfigure(0, weight=1)  # Give the row weight

        # Notebook Widget
        self.epoch_notebook = ttk.Notebook(self.report_content_frame)
        # *** Place Notebook in Center Column ***
        self.epoch_notebook.grid(row=0, column=1, sticky="nsew")

        # Create Dictionary to Store Year/Epoch Frames
        self.epoch_frames = {}

        # Create only the first Epoch/Year Frame (Year 1)
        self.create_year_frame(0)

        # Initialize Epoch/Year Counter and Label
        if self.epochs.get() > 0:
            self.current_epoch = 0
            self.epoch_counter.config(text=f"Year {self.current_epoch + 1}/{self.epochs.get()}")
            self.report_viz_frame.config(text=f"Year {self.current_epoch + 1}'s Report Cards")

        # Bind Configure Event to Update Scroll Region
        self.report_content_frame.bind(
            "<Configure>",
            lambda e: self.report_canvas.configure(scrollregion=self.report_canvas.bbox("all"))
        )

    def create_year_frame(self, epoch):
        """Creates a frame for the specified epoch/year and adds it to the epoch_notebook."""

        # 7.1 Create a Frame for the Year/Epoch
        frame = tk.Frame(self.epoch_notebook)  # Add to epoch_notebook, NOT self.notebook

        # 7.2 Add the Frame as a Tab to the Notebook
        self.epoch_notebook.add(frame, text=f"Year {epoch + 1}")

        # 7.3 Store the Frame in the epoch_frames Dictionary
        self.epoch_frames[epoch] = frame

        # 7.4 Configure Grid Layout for the Frame (4 rows for Term Cards)
        for i in range(4):
            frame.grid_rowconfigure(i, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # *** Center Alignment ***
        frame.grid_columnconfigure(1, weight=1)  # Add a new column with equal weight
        frame.grid_columnconfigure(2, weight=1)  # Add a new column with equal weight

    def create_sample_output_frame(self, parent):
        # Create LabelFrame for Sample Output
        self.sample_output_frame = tk.LabelFrame(
            parent,
            text="Examination Result",
            width=400,
            height=100,
        )
        self.sample_output_frame.grid(row=1, column=0, sticky="ew")
        self.sample_output_frame.grid_propagate(False)

        # Create Scrollbar Frame
        sample_output_scrollbar_frame = tk.Frame(self.sample_output_frame, width=20)
        sample_output_scrollbar_frame.grid(row=0, column=1, sticky="ns")

        # Create Vertical Scrollbar
        sample_output_scrollbar = tk.Scrollbar(
            sample_output_scrollbar_frame,
            orient="vertical"
        )
        sample_output_scrollbar.pack(fill=tk.Y, expand=True)

        # Create Sample Output Text Widget
        self.sample_output_text = tk.Text(
            self.sample_output_frame,
            wrap=tk.WORD,
            yscrollcommand=sample_output_scrollbar.set
        )
        self.sample_output_text.grid(row=0, column=0, sticky="nsew")

        # Configure Scrollbar Command
        sample_output_scrollbar.config(command=self.sample_output_text.yview)

    def create_console_frame(self, parent):
        # Create Console Frame
        self.console_frame = tk.Frame(parent, width=400, height=100)
        self.console_frame.grid(row=2, column=0, sticky="ew")
        self.console_frame.grid_propagate(False)

        # Create Console Log Frame
        self.console_log_frame = tk.Frame(self.console_frame)
        self.console_log_frame.grid(row=0, column=0, sticky="nsew")

        # Create Console Scrollbar Frame
        console_scrollbar_frame = tk.Frame(self.console_log_frame, width=20)
        console_scrollbar_frame.grid(row=0, column=1, sticky="ns")

        # Create Vertical Scrollbar
        console_scrollbar = tk.Scrollbar(
            console_scrollbar_frame,
            orient="vertical"
        )
        console_scrollbar.pack(fill=tk.Y, expand=True)

        # Create Console Box Text Widget
        self.console_box = tk.Text(
            self.console_log_frame,
            wrap="word",
            yscrollcommand=console_scrollbar.set
        )
        self.console_box.grid(row=0, column=0, sticky="nsew")

        # Configure Scrollbar Command
        console_scrollbar.config(command=self.console_box.yview)

    def create_input_output_frame(self, parent):
        # Create Outer Frame (to control width)
        outer_frame = tk.Frame(parent, width=400)
        outer_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create LabelFrame for Input/Output
        input_output_frame = tk.LabelFrame(outer_frame, text="Input/Output")
        input_output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create Model Directory Selection Frame
        model_dir_frame = tk.Frame(input_output_frame)
        model_dir_frame.pack(fill=tk.BOTH)

        # Create "Input Model:" Label
        tk.Label(model_dir_frame, text="Input Model:", width=15).pack(side=tk.LEFT)

        # Create Entry for Model Directory
        tk.Entry(model_dir_frame, textvariable=self.model_dir, width=73).pack(side=tk.LEFT)

        # Create "Browse" Button for Model Directory
        tk.Button(model_dir_frame, text="Browse", command=self.browse_model_dir).pack(side=tk.LEFT)

        # Create Output Directory Selection Frame
        output_dir_frame = tk.Frame(input_output_frame)
        output_dir_frame.pack(fill=tk.BOTH)

        # Create "Output Model:" Label
        tk.Label(output_dir_frame, text="Output Model:", width=15).pack(side=tk.LEFT)

        # Create Entry for Output Directory
        tk.Entry(output_dir_frame, textvariable=self.output_dir, width=73).pack(side=tk.LEFT)

        # Create "Browse" Button for Output Directory
        tk.Button(output_dir_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.LEFT)

    def browse_model_dir(self):
        # Open File Dialog to Choose a Directory
        directory = filedialog.askdirectory()

        # If a Directory is Selected, Set the model_dir Variable
        if directory:
            self.model_dir.set(directory)

    def browse_output_dir(self):
        # Open File Dialog to Choose a Directory
        directory = filedialog.askdirectory()

        # If a Directory is Selected, Set the output_dir Variable
        if directory:
            self.output_dir.set(directory)

    def create_education_frame(self, parent):
        # Create LabelFrame for Education Parameters
        education_frame = tk.LabelFrame(parent, text="Education")
        education_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create "Years:" Label
        year_label = tk.Label(education_frame, text="Years:")
        year_label.grid(row=0, column=0, sticky="w", padx=(0, 5))

        # Create Entry for Years (Epochs)
        self.years_entry = tk.Entry(education_frame, textvariable=self.epochs, width=86)
        self.years_entry.grid(row=0, column=1)

        # Create "Batch Size:" Label
        batch_label = tk.Label(education_frame, text="Batch Size:")
        batch_label.grid(row=1, column=0, sticky="w", padx=(0, 5))

        # Create Entry for Batch Size
        self.batch_size_entry = tk.Entry(education_frame, textvariable=self.batch_size, width=86)
        self.batch_size_entry.grid(row=1, column=1)

        # Create "Learning Rate:" Label
        learning_rate_label = tk.Label(education_frame, text="Learning Rate:")
        learning_rate_label.grid(row=2, column=0, sticky="w", padx=(0, 5))

        # Create Entry for Learning Rate
        self.learning_rate_entry = tk.Entry(education_frame, textvariable=self.learning_rate, width=86)
        self.learning_rate_entry.grid(row=2, column=1)

        # Create "Modules per Term:" Label
        inputs_label = tk.Label(education_frame, text="Modules per Term:")
        inputs_label.grid(row=3, column=0, sticky="w", padx=(0, 5))

        # Create Entry for Inputs per Mini-Epoch
        self.inputs_per_miniepoch_entry = tk.Entry(education_frame, textvariable=self.inputs_per_miniepoch, width=83)
        self.inputs_per_miniepoch_entry.grid(row=3, column=1)

    def create_personalisation_frame(self, parent):
        # Create LabelFrame for Personalization Options
        personalisation_frame = tk.LabelFrame(parent, text="Personalisation")
        personalisation_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create "Your Name:" Label
        tk.Label(personalisation_frame, text="Your Name:").grid(row=0, column=0)

        # Create Entry for Your Name
        tk.Entry(personalisation_frame, textvariable=self.your_name, width=93).grid(row=0, column=1)

        # Create "DEI's Name:" Label
        tk.Label(personalisation_frame, text="DEI's Name:").grid(row=1, column=0)

        # Create Entry for DEI's Name
        tk.Entry(personalisation_frame, textvariable=self.model_name, width=93).grid(row=1, column=1)

    def create_test_prompt_frame(self, parent):
        # Create Outer Frame (to control width)
        outer_frame = tk.Frame(parent, width=400)
        outer_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create LabelFrame for Test Prompt
        test_prompt_frame = tk.LabelFrame(outer_frame, text="Examination Question")
        test_prompt_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create Instruction Label for Test Prompt
        tk.Label(
            test_prompt_frame,
            text="Ask DEI questions while it is spawning to gauge educational and contextual growth:",
            wraplength=400
        ).grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Create Text Entry Widget for Test Prompt
        self.test_prompt_entry = tk.Text(test_prompt_frame, width=77, height=3)
        self.test_prompt_entry.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Set Default Text in Test Prompt Entry
        self.test_prompt_entry.insert(tk.END, "Hello! Tell me about yourself?")

    def create_datasets_frame(self, parent):
        # Create LabelFrame for Datasets
        datasets_frame = tk.LabelFrame(parent, text="Courses")
        datasets_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # Create "Number of Courses:" Label
        tk.Label(datasets_frame, text="Number of Courses:").grid(row=0, column=0)

        # Create Spinbox to Control Number of Courses
        tk.Spinbox(
            datasets_frame,
            from_=1,
            to=60,
            textvariable=self.num_datasets,
            width=5,
            command=self.update_dataset_entries
        ).grid(row=0, column=1)

        # Create Frame to Hold Dataset Entry Widgets
        entries_frame = tk.Frame(datasets_frame)
        entries_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        datasets_frame.grid_columnconfigure(0, minsize=600)

        # Create Frame to Hold Scrollbar
        dataset_scrollbar_frame = tk.Frame(entries_frame, width=50)
        dataset_scrollbar_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create Vertical Scrollbar
        dataset_scrollbar = ttk.Scrollbar(dataset_scrollbar_frame, orient="vertical")
        dataset_scrollbar.pack(fill=tk.Y, expand=True)

        # Create Canvas to Hold Dataset Entries
        self.dataset_canvas = tk.Canvas(
            entries_frame,
            yscrollcommand=dataset_scrollbar.set,
            height=130,
            width=600
        )
        self.dataset_canvas.pack(side=tk.LEFT, fill=tk.BOTH)

        # Configure Scrollbar Command
        dataset_scrollbar.config(command=self.dataset_canvas.yview)

        # Create Inner Frame for Dataset Entries
        self.dataset_inner_frame = tk.Frame(self.dataset_canvas)
        self.dataset_canvas.create_window((0, 0), window=self.dataset_inner_frame, anchor="nw")

        # Create Entry Widgets and Browse Buttons
        for i in range(60):
            # Create Frame for Each Entry
            entry_frame = tk.Frame(self.dataset_inner_frame)
            entry_frame.pack(fill=tk.X)

            # Create "Course X:" Label
            tk.Label(entry_frame, text=f"Course {i + 1}:").pack(side=tk.LEFT)

            # Create Entry for Dataset Path
            entry = tk.Entry(entry_frame, width=80)
            entry.pack(side=tk.LEFT)

            # Create "Browse" Button
            browse_button = tk.Button(
                entry_frame,
                text="Browse",
                command=lambda idx=i: self.browse_dataset(idx)
            )
            browse_button.pack(side=tk.LEFT, padx=2)

            # Store Entry and Button
            self.dataset_entries.append(entry)
            self.dataset_browse_buttons.append(browse_button)

        # Load Existing Courses from "courses" Directory
        datasets_dir = os.path.join(os.getcwd(), "courses")
        dataset_files = [
            os.path.join(datasets_dir, f)
            for f in os.listdir(datasets_dir)
            if os.path.isfile(os.path.join(datasets_dir, f)) and f.endswith(".csv")
        ]

        # Populate Entries with Existing Courses
        for i, file_path in enumerate(dataset_files):
            if i < len(self.dataset_entries):
                self.dataset_entries[i].insert(0, file_path)

        # Update Entry State Based on Number of Courses
        self.update_dataset_entries()

        # Configure Canvas to Update Scroll Region on Resize
        self.dataset_canvas.bind("<Configure>", lambda e: self.dataset_canvas.configure(scrollregion=self.dataset_canvas.bbox("all")))

    def browse_dataset(self, index):
        # Open File Dialog to Choose a CSV File
        filepath = filedialog.askopenfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )

        # If a File is Selected, Update the Corresponding Entry
        if filepath:
            self.dataset_entries[index].delete(0, tk.END)
            self.dataset_entries[index].insert(0, filepath)

    def create_buttons_frame(self, parent):
        # Create Frame for Control Buttons
        self.buttons_frame = tk.Frame(parent)
        self.buttons_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Create "Birth" Button (Start Training)
        self.start_training_button = tk.Button(
            self.buttons_frame, text="Birth", command=self.start_training
        )
        self.start_training_button.pack(side=tk.LEFT, padx=5)

        # Create "Abort" Button (Stop Training)
        self.stop_training_button = tk.Button(
            self.buttons_frame,
            text="Abort",
            command=self.stop_training,
            state="disabled"  # Initially disabled as no training is running
        )
        self.stop_training_button.pack(side=tk.LEFT, padx=5)

        # Create "Update Test" Button
        self.update_test_prompt_button = tk.Button(
            self.buttons_frame,
            text="Update Test",
            command=self.update_test_prompt
        )
        self.update_test_prompt_button.pack(side=tk.LEFT, padx=5)

        # Create "Clear History" Button
        self.clear_console_button = tk.Button(
            self.buttons_frame,
            text="Clear History",
            command=self.clear_console
        )
        self.clear_console_button.pack(side=tk.LEFT, padx=5)

    def clean_name(self, name):
        """Cleans a name string to make it safe for filenames."""

        # Replace Invalid Characters with Underscores
        name = re.sub(r'[\\/:*?"<>|]', "_", name)

        # Replace Spaces with Underscores
        name = name.replace(" ", "_")

        # Return the Cleaned Name
        return name

    def start_training(self):
        """Begins the DEI training process."""
        # 1. Access global variables (used during training)
        global model, tokenizer, device, datasets, dataset_weights, person_name, model_name, test_prompt

        # 2. Input Validation: Check if a model directory is selected
        if not self.model_dir.get():
            messagebox.showerror("Error", "Please select a model directory.")
            return

        # 3. Disable buttons during training to prevent unwanted interactions
        self.start_training_button.config(state="disabled")
        self.stop_training_button.config(state="normal")
        self.clear_console_button.config(state="disabled")
        self.previous_epoch_button.config(state="disabled")
        self.next_epoch_button.config(state="disabled")

        # 4. Initialize/Clean Global Variables
        person_name = self.clean_name(self.your_name.get())  # Clean the person's name
        model_name = self.clean_name(self.model_name.get())  # Clean the model's name
        test_prompt = self.test_prompt_entry.get("1.0", tk.END).strip()  # Get the test prompt from the text entry
        datasets = []  # Initialize empty list to store datasets
        dataset_weights = [1.0] * self.num_datasets.get()  # Initialize dataset weights - all courses start with equal weight

        # 5. Load Datasets from Selected CSV Files
        for i in range(self.num_datasets.get()):
            file_path = self.dataset_entries[i].get()  # Get the path from the dataset entry field
            if file_path:  # Check if a path is provided
                try:
                    # Try loading the dataset
                    dataset = load_dataset(file_path, person_name, model_name)
                    if dataset:
                        # If dataset loads successfully, add it to the datasets list
                        datasets.append(dataset)
                    else:
                        # If loading fails, display a warning and set the weight for this course to 0
                        messagebox.showwarning("Warning", f"Failed to load Course {i + 1}. Continuing without it.")
                        dataset_weights[i] = 0.0
                except Exception as e:
                    # Handle any exception that occurs during dataset loading
                    messagebox.showerror("Error", f"Failed to load Course {file_path}: {str(e)}")
                    return

        # 6. Validate Training Parameters (make sure they are the correct data type)
        try:
            self.epochs.set(int(self.epochs.get()))  # Convert years to integer
            self.batch_size.set(int(self.batch_size.get()))  # Convert batch size to integer
            self.learning_rate.set(float(self.learning_rate.get()))  # Convert learning rate to float
            self.inputs_per_miniepoch.set(int(self.inputs_per_miniepoch.get()))  # Convert modules per term to integer
        except ValueError as e:
            # Display an error message if any of the conversions fail
            messagebox.showerror("Error", f"Invalid training parameter: {str(e)}")
            return

        # 7. Load the Model and Tokenizer
        if self.model_handler.load_model(self.model_dir.get()):
            model = self.model_handler.model
            tokenizer = self.model_handler.tokenizer

            # 8. Add Special Tokens (if any) and Resize Model Embeddings
            special_tokens = ["[YOUR_SPECIAL_TOKEN1]", "[YOUR_SPECIAL_TOKEN2]"]  # Add your actual special tokens here
            tokenizer.add_tokens(special_tokens)  # Add special tokens to the tokenizer
            model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to accommodate new tokens

            # 9. Get the Device (CPU or GPU) and Move Model to Device
            device = get_device()
            model.to(device)

            # 10. Reset Data Storage for the New Training Run
            self.epoch_data = {}  # Initialize a dictionary to store data for each epoch and mini-epoch
            self.sample_outputs = []  # Clear the sample outputs list

            # 11. Create the FIRST Epoch Frame (Year 1) when training STARTS
            self.epoch_notebook.select(0)
            self.update_epoch_counter_and_label()

            # 12. Start the Training Thread
            self.training_thread = threading.Thread(target=self.run_training_thread)
            self.training_thread.start()

            # 13. Start GUI Update Loops
            self.process_gui_updates()  # Start the process that handles GUI updates from the training thread
            self.process_console_updates()  # Start the process that handles console updates from the training thread

        else:
            # 14. Handle Model Loading Failure
            messagebox.showerror("Error", "Failed to load model")
            # Re-enable the buttons
            self.start_training_button.config(state="normal")
            self.stop_training_button.config(state="disabled")
            self.clear_console_button.config(state="normal")

    def run_training_thread(self):
        """Runs the training process in a separate thread."""

        # Access Global Variables
        global model, tokenizer, device, datasets, dataset_weights, person_name, model_name, test_prompt

        # Redirect Standard Output to the Console
        sys.stdout = self.redirect_stdout

        # Log Start of Training
        logging.info("Starting training thread")
        try:
            # Clear the Stop Training Event Flag
            self.stop_training_event.clear()

            # Start the Education (Training) Process
            educate(
                self,  # Reference to the CreationWindow instance
                model,  # Loaded model
                tokenizer,  # Loaded tokenizer
                self.epochs.get(),  # Number of epochs
                self.inputs_per_miniepoch.get(),  # Inputs per mini-epoch
                datasets,  # Loaded datasets
                dataset_weights,  # Dataset weights
                person_name,  # Person's name
                model_name,  # Model's name
                test_prompt,  # Test prompt
                self.gui_update_queue,  # GUI update queue
                self.console_update_queue,  # Console update queue
                self.master,  # Root Tkinter window
                device  # Device (CPU or GPU)
            )

        except Exception as e:
            # Handle Exceptions During Training
            self.master.after(0,
                              lambda: self.main_app.output_queue.put((f"Error during training: {e}\n", "console",
                                                                     "Creation")))
            self.output_queue.put(
                (
                    lambda: messagebox.showerror(
                        "Error", f"Error during training: {e}"
                    ),
                    "console",
                    "Creation",
                )
            )
        finally:
            # Signal Training Completion
            self.gui_update_queue.put({"training_complete": True})

    def process_gui_updates(self):
        """Processes updates for GUI elements during training."""
        try:
            # 1. Set a maximum number of iterations to prevent an infinite loop.
            max_iterations = 100

            # 2. Initialize an iteration counter.
            iteration_count = 0

            # 3. Start a loop that continues as long as:
            #    - The iteration count is less than the maximum iterations.
            #    - The GUI update queue is not empty (there are updates to process).
            while iteration_count < max_iterations and not self.gui_update_queue.empty():
                # 4. Get the next data item from the GUI update queue.
                data = self.gui_update_queue.get()

                # 5. Check if the data indicates that training is complete.
                if "training_complete" in data:
                    # 5.1 If training is complete, call the finish_training function.
                    self.finish_training()

                # 6. Check if the data indicates that a mini-epoch (term) is complete.
                elif "MINI_EPOCH_COMPLETE" in data:
                    # 6.1 Get the epoch number and mini-epoch number from the data.
                    epoch = data["MINI_EPOCH_COMPLETE"] - 1  # Adjust for zero-based indexing
                    mini_epoch = data["mini_epoch"]

                    # 6.2 Acquire the lock to access the shared epoch_data dictionary safely.
                    with self.term_frames_lock:
                        # 6.3 Check if the epoch_data for this epoch has all 4 mini-epochs (terms) completed.
                        if epoch in self.epoch_data and len(self.epoch_data[epoch]) == 4:
                            # 6.3.1 If all 4 terms are complete, call update_epoch_term_cards to create the term cards.
                            self.update_epoch_term_cards(epoch)

                # 7. Check if the data indicates a regular mini-epoch (term) update.
                elif "mini_epoch" in data:
                    # 7.1 Get epoch and mini_epoch information from the data.
                    epoch = data["epoch"] - 1  # Adjust for zero-based indexing
                    mini_epoch = data["mini_epoch"]

                    # 7.2 Update the sub-module progress bar based on the data.
                    sub_module_counter = data.get("sub_module_counter", 0)
                    total_sub_modules = data.get("total_sub_modules", 1)
                    self.update_sub_module_progress(sub_module_counter, total_sub_modules)

                    # 7.3 Update the current epoch counter and the main progress bars.
                    self.current_epoch = epoch
                    self.year_progress.step(100 / (self.epochs.get() * 4))
                    term_progress = data.get("term_progress", 0)
                    self.term_progress["value"] = term_progress

                    # 7.4 Update the sample output display with the new sample output from the data.
                    sample_output = data.get("sample_output", "")
                    if sample_output:
                        self.sample_outputs.insert(0, sample_output)
                        self.update_sample_output()

                    # 7.5 Update the epoch_data dictionary with the term data, using the lock for thread safety.
                    with self.term_frames_lock:
                        if epoch not in self.epoch_data:
                            self.epoch_data[epoch] = {}

                        # Create a dictionary for the term data.
                        term_data = {
                            "knowledge_radius": self.evaluation.knowledge_radius,
                            "understanding_depth": self.evaluation.understanding_depth,
                            "improvement_momentum": self.evaluation.improvement_momentum,
                            "report": self.evaluation.generate_report(epoch + 1, mini_epoch),
                            "sample_output": sample_output
                        }

                        # Store the term data in the epoch_data dictionary.
                        self.epoch_data[epoch][mini_epoch] = term_data

                    # 7.6 Update the epochs_completed counter.
                    self.epochs_completed = max(self.epochs_completed, epoch + 1)

                # 8. Increment the iteration counter.
                iteration_count += 1

            # 9. Schedule the process_gui_updates function to be called again after 100 milliseconds.
            self.master.after(100, self.process_gui_updates)
            return

        except Exception as e:
            # 10. Handle any exceptions that occur during GUI updates.
            print(f"Error processing GUI updates: {e}")

    def process_console_updates(self):
        """Processes updates for the console box."""
        try:
            while not self.console_update_queue.empty():
                message, target, window_name = self.console_update_queue.get()

                if target == "console" and window_name == "Creation":
                    self.console_box.config(state="normal")  # Allow text insertion
                    self.console_box.insert(tk.END, message)  # Insert message
                    self.console_box.see(tk.END)  # Scroll to the end
                    self.console_box.config(state="disabled")  # Prevent further editing
        except Exception as e:
            print(f"Error processing console updates: {e}")
        finally:
            # Schedule Next Console Update Check
            self.master.after(10, self.process_console_updates)

    def finish_training(self):
        """Handles GUI updates and notifications after training."""

        # Ensure term cards for the last epoch are updated:
        with self.term_frames_lock:
            self.update_epoch_term_cards(self.epochs.get())  # Update for the last epoch

        # Re-enable Buttons
        self.start_training_button.config(state="normal")
        self.stop_training_button.config(state="disabled")
        self.clear_console_button.config(state="normal")
        self.previous_epoch_button.config(state="normal")
        self.next_epoch_button.config(state="normal")

        if not self.stop_training_event.is_set():
            messagebox.showinfo(
                "New Life Spawned!",
                "Go to the Interaction window and load it from the educated models directory and see what it has to say!"
            )

        # Reset Training Thread
        self.training_thread = None

    def stop_training(self):
        """Stops the training process."""

        # Check if Training Thread is Running
        if self.training_thread:
            self.stop_training_event.set()  # Signal thread to stop
            self.training_thread = None  # Reset training thread

        # Reset GUI Buttons
        self.start_training_button.config(state="normal")
        self.stop_training_button.config(state="disabled")
        self.clear_console_button.config(state="normal")
        self.previous_epoch_button.config(state="normal")
        self.next_epoch_button.config(state="normal")

    def create_term_frame(self, term_index, term_data, parent=None):
        """Creates a Term Card frame and populates it with data."""

        # Create the Term Card frame
        term_frame = tk.LabelFrame(parent, text=f"Term {term_index}")

        # Set the column width for labels
        column_width = 380

        # Create frames for labels within the Term Card
        left_label_frame = tk.Frame(term_frame, width=column_width)
        left_label_frame.pack(side=tk.LEFT, fill="both", expand=True)

        report_label_frame = tk.Frame(term_frame, width=column_width)
        report_label_frame.pack(side=tk.LEFT, fill="both", expand=True)

        # Create labels for knowledge radius, understanding depth, improvement momentum, and report
        knowledge_radius_label = tk.Label(
            left_label_frame, text=f"Knowledge Radius: {term_data.get('knowledge_radius', 'N/A')}",
            wraplength=360, justify="left"
        )
        knowledge_radius_label.pack(fill="both", expand=True)

        understanding_depth_label = tk.Label(
            left_label_frame, text=f"Understanding Depth: {term_data.get('understanding_depth', 'N/A')}",
            wraplength=360, justify="left"
        )
        understanding_depth_label.pack(fill="both", expand=True)

        improvement_momentum_label = tk.Label(
            left_label_frame, text=f"Improvement Momentum: {term_data.get('improvement_momentum', 'N/A')}",
            wraplength=360, justify="left"
        )
        improvement_momentum_label.pack(fill="both", expand=True)

        report_label = tk.Label(
            report_label_frame, text=f"Report: {term_data.get('report', 'Report of Term X')}",
            wraplength=360, justify="left"
        )
        report_label.pack(fill="both", expand=True)

        return term_frame

    def load_dataset(self, path, person_name=None, model_name=None):
        """Loads a dataset from a CSV file."""
        try:
            # Read CSV File into a Pandas DataFrame
            df = pd.read_csv(path, names=["input", "output"])

            # Replace Placeholders with Provided Names
            if person_name:
                person_name = self.clean_name(person_name)
                df["input"] = df["input"].str.replace("Conrad", person_name)
                df["output"] = df["output"].str.replace("Conrad", person_name)
            if model_name:
                model_name = self.clean_name(model_name)
                df["input"] = df["input"].str.replace("Logos", model_name)
                df["output"] = df["output"].str.replace("Logos", model_name)

            # Fill Missing Values with "[No Response]"
            df.fillna("[No Response]", inplace=True)

            # Convert DataFrame to List of Lists and Return
            return df.values.tolist()

        except Exception as e:
            # Handle Exceptions During Dataset Loading
            print(f"Failed to load Course {path}: {str(e)}")
            return None

    def update_dataset_entries(self):
        """Updates the state of dataset entry widgets based on the selected number of courses."""

        # Get the Number of Datasets from the Spinbox
        num_datasets = self.num_datasets.get()

        # Iterate through Dataset Entry Widgets and Browse Buttons
        for i in range(60):
            if i < num_datasets:
                self.dataset_entries[i].config(state="normal")  # Enable entry and browse button
                self.dataset_browse_buttons[i].config(state="normal")
            else:
                self.dataset_entries[i].config(state="disabled")  # Disable entry and browse button
                self.dataset_browse_buttons[i].config(state="disabled")

    def clear_console(self):
        """Clears the console box."""
        self.console_box.delete("1.0", tk.END)  # Delete all text in the console box

    def next_epoch(self):
        """Navigates to the next epoch."""
        # 1. Get the index of the currently selected tab
        current_tab_index = self.epoch_notebook.index(self.epoch_notebook.select())

        # 2. Check if there's a next tab to navigate to
        if current_tab_index < self.epochs.get() - 1:  # Changed from self.current_epoch to current_tab_index
            # 3. Increment the current tab index
            next_tab_index = current_tab_index + 1

            # 4. Create the Year Frame if it doesn't exist
            if next_tab_index > 0 and next_tab_index not in self.epoch_frames:
                self.create_year_frame(next_tab_index)

            # 5. Select the next tab
            self.epoch_notebook.select(next_tab_index)

            # 6. Update the current epoch counter to match the selected tab
            self.current_epoch = next_tab_index

            # 7. Update the navigation buttons (enable/disable)
            self.update_navigation_buttons()

            # 8. Update the epoch counter label and the report frame label
            self.update_epoch_counter_and_label()

    def previous_epoch(self):
        """Navigates to the previous epoch."""

        # Check if there's a previous epoch to navigate to (current epoch is greater than 0)
        if self.current_epoch > 0:
            # Decrement the current epoch counter
            self.current_epoch -= 1

            # Select the notebook tab for the current epoch
            self.epoch_notebook.select(self.current_epoch)  # Changed from self.notebook to self.epoch_notebook

            # Update the navigation buttons (enable/disable)
            self.update_navigation_buttons()

            # Update the epoch counter label and the report frame label
            self.update_epoch_counter_and_label()

    def update_epoch_term_cards(self, epoch):
        """Updates all 4 Term Card frames for the given epoch."""
        print(f"Updating term cards for epoch: {epoch}")  # Debugging
        print(f"epoch_data: {self.epoch_data}")  # Debugging
        if epoch in self.epoch_data:
            with self.term_frames_lock:
                for mini_epoch in range(4):
                    print(f"mini_epoch: {mini_epoch}")  # Debugging
                    print(f"Keys in epoch_data[{epoch}]: {self.epoch_data[epoch].keys()}")  # Debugging
                    term_data = self.epoch_data[epoch].get(mini_epoch)
                    if term_data:
                        frame = self.epoch_frames[epoch]
                        term_frame = self.create_term_frame(mini_epoch + 1, term_data, parent=frame)
                        term_frame.grid(row=mini_epoch, column=0, sticky="nsew", padx=5, pady=5)

    def update_navigation_buttons(self):
        """Updates the state of the navigation buttons."""
        if self.current_epoch == self.epochs.get() - 1:  # Disable next button if at the last epoch
            self.next_epoch_button.config(state="disabled")
        else:
            self.next_epoch_button.config(state="normal")  # Enable next button

        if self.current_epoch == 0:  # Disable previous button if at the first epoch
            self.previous_epoch_button.config(state="disabled")
        else:
            self.previous_epoch_button.config(state="normal")  # Enable previous button

    def update_epoch_counter_and_label(self):
        """Updates the epoch counter label and the report frame label."""
        # Update Epoch Counter Label
        self.epoch_counter.config(text=f"Year {self.current_epoch + 1}/{self.epochs.get()}")

        # Update Report Frame Label
        self.report_viz_frame.config(text=f"Year {self.current_epoch + 1}'s Report Cards")

    def update_sub_module_progress(self, sub_module_counter, total_sub_modules):
        """Updates the sub-module progress bar."""
        if total_sub_modules > 0:
            progress = (sub_module_counter / total_sub_modules) * 100
            self.sub_module_progress["value"] = progress
        else:
            self.sub_module_progress["value"] = 0  # Reset progress bar

    def update_sample_output(self):
        """Updates the sample output display."""
        self.sample_output_text.configure(state="normal")  # Allow text editing
        self.sample_output_text.delete("1.0", tk.END)  # Clear previous output
        for output in self.sample_outputs:
            self.sample_output_text.insert(tk.END, f"{output}\n")  # Insert new sample outputs
        self.sample_output_text.yview_moveto(0.0)  # Scroll to the top
        self.sample_output_text.configure(state="disabled")  # Prevent editing

    def update_test_prompt(self):
        """Updates the global test prompt."""
        global test_prompt
        test_prompt = self.test_prompt_entry.get("1.0", tk.END).strip()
        print(f"Examination Question Updated to: {test_prompt}")

    def update_term_card(self, epoch, mini_epoch, term_data):
        """Creates and populates a Term Card on the correct year/epoch frame."""
        if epoch in self.epoch_frames:
            frame = self.epoch_frames[epoch]
            term_frame = self.create_term_frame(mini_epoch + 1, term_data, parent=frame)
            term_frame.grid(row=mini_epoch, column=1, sticky="nsew", padx=5, pady=5)  # Place in the center column

    def process_gui_updates(self):
        """Processes updates for GUI elements during training."""
        try:
            # 1. Set a maximum number of iterations to prevent an infinite loop.
            max_iterations = 100
            iteration_count = 0

            # 2. Start a loop that continues as long as:
            #    - The iteration count is less than the maximum iterations.
            #    - The GUI update queue is not empty (there are updates to process).
            while iteration_count < max_iterations and not self.gui_update_queue.empty():
                # 3. Get the next data item from the GUI update queue.
                data = self.gui_update_queue.get()

                # 4. Check if the data indicates that training is complete.
                if "training_complete" in data:
                    # 4.1 If training is complete, call the finish_training function.
                    self.finish_training()

                # 5. Check if the data indicates that a mini-epoch (term) is complete.
                elif "MINI_EPOCH_COMPLETE" in data:
                    # 5.1 Get the epoch number and mini-epoch number from the data.
                    epoch = data["MINI_EPOCH_COMPLETE"] - 1  # Adjust for zero-based indexing
                    mini_epoch = data["mini_epoch"]

                    # 5.2 Acquire the lock to access the shared epoch_data dictionary safely.
                    with self.term_frames_lock:
                        # 5.3 Check if data for this epoch and mini-epoch exists.
                        if epoch in self.epoch_data and mini_epoch in self.epoch_data[epoch]:
                            # 5.4 Get the term data for this epoch and mini-epoch
                            term_data = self.epoch_data[epoch][mini_epoch]

                            # 5.5 Get the frame for the corresponding epoch (year)
                            frame = self.epoch_frames[epoch]

                            # 5.6 Create the Term Card frame and place it on the grid
                            term_frame = self.create_term_frame(mini_epoch + 1, term_data, parent=frame)
                            term_frame.grid(row=mini_epoch, column=0, sticky="nsew", padx=5, pady=5)

                # 6. Check if the data indicates a regular mini-epoch (term) update.
                elif "mini_epoch" in data:
                    # 6.1 Get epoch and mini_epoch information from the data.
                    epoch = data["epoch"] - 1  # Adjust for zero-based indexing
                    mini_epoch = data["mini_epoch"]

                    # 6.2 Update the sub-module progress bar based on the data.
                    sub_module_counter = data.get("sub_module_counter", 0)
                    total_sub_modules = data.get("total_sub_modules", 1)
                    self.update_sub_module_progress(sub_module_counter, total_sub_modules)

                    # 6.3 Update the current epoch counter and the main progress bars.
                    self.current_epoch = epoch
                    self.year_progress.step(100 / (self.epochs.get() * 4))
                    term_progress = data.get("term_progress", 0)
                    self.term_progress["value"] = term_progress

                    # 6.4 Update the sample output display with the new sample output from the data.
                    sample_output = data.get("sample_output", "")
                    if sample_output:
                        self.sample_outputs.insert(0, sample_output)
                        self.update_sample_output()

                    # 6.5 Update the epoch_data dictionary with the term data, using the lock for thread safety.
                    with self.term_frames_lock:
                        if epoch not in self.epoch_data:
                            self.epoch_data[epoch] = {}

                        # Create a dictionary for the term data.
                        term_data = {
                            "knowledge_radius": self.evaluation.knowledge_radius,
                            "understanding_depth": self.evaluation.understanding_depth,
                            "improvement_momentum": self.evaluation.improvement_momentum,
                            "report": self.evaluation.generate_report(epoch + 1, mini_epoch),
                            "sample_output": sample_output
                        }

                        # Store the term data in the epoch_data dictionary.
                        self.epoch_data[epoch][mini_epoch] = term_data

                    # 6.6 Update the epochs_completed counter.
                    self.epochs_completed = max(self.epochs_completed, epoch + 1)

                # 7. Increment the iteration counter.
                iteration_count += 1

            # 8. Schedule the process_gui_updates function to be called again after 100 milliseconds.
            self.master.after(100, self.process_gui_updates)
            return

        except Exception as e:
            # 9. Handle any exceptions that occur during GUI updates.
            print(f"Error processing GUI updates: {e}")

# ----------------------
# Interaction Window
# ----------------------

class InteractionWindow(BaseApplicationWindow):
    """Window for interacting with a trained DEI model."""

    def __init__(
        self, master, main_app, model=None, tokenizer=None, notebook=None
    ):
        """Initializes the InteractionWindow."""
        self.output_queue = main_app.output_queue
        self.notebook = notebook  # Store the notebook
        self.redirect_stdout = main_app.redirect_stdout_dict["Interaction"]  # Correctly access redirect_stdout_dict
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = "DEI" if self.model else None
        self.model_dir = os.path.join(".", "models", "educated-models")
        super().__init__(master, main_app, notebook=notebook)
        self.load_model()  # Attempt to load a trained model
        if self.model and self.tokenizer:
            self.model_name = self.model.__class__.__name__
            self.input_entry.config(state="normal")
            self.send_button.config(state="normal")
        tk.Label(self.model_path_frame, text="Load DEI").grid(
            row=0, column=0, sticky="w"
        )
        self.browse_button = tk.Button(
            self.model_path_frame, text="Browse", command=self.browse_model
        )
        self.browse_button.grid(row=0, column=2, sticky="w")
        tk.Label(self.model_path_frame, text="Load DEI").grid(
            row=0, column=0, sticky="w"
        )

    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_frame = tk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=0)
        main_frame.grid_rowconfigure(2, weight=0)
        main_frame.grid_rowconfigure(3, weight=0)  # Row for the progress bar
        main_frame.grid_rowconfigure(4, weight=0)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)  # Add a new column for the parameters frame

        dialogue_frame = tk.Frame(main_frame)
        dialogue_frame.grid(row=0, column=0, sticky="nsew")
        dialogue_frame.grid_rowconfigure(0, weight=1)
        dialogue_frame.grid_columnconfigure(0, weight=1)

        self.dialogue_box = scrolledtext.ScrolledText(
            dialogue_frame, wrap=tk.WORD, width=100, height=20
        )
        self.dialogue_box.grid(row=0, column=0, sticky="nsew")

        memory_console_frame = tk.Frame(main_frame)
        memory_console_frame.grid(row=1, column=0, sticky="ew")
        memory_console_frame.grid_rowconfigure(0, weight=1)
        memory_console_frame.grid_columnconfigure(0, weight=1)
        memory_console_frame.grid_columnconfigure(1, weight=1)

        memory_frame = tk.Frame(memory_console_frame)
        memory_frame.grid(row=0, column=0, sticky="nsew")
        self.physical_memory_label = tk.Label(
            memory_frame, text="Physical Memory: 0.0 GB / 0.0 GB"
        )
        self.physical_memory_label.pack(pady=5)
        self.swap_memory_label = tk.Label(
            memory_frame, text="Swap Memory: 0.0 GB / 0.0 GB"
        )
        self.swap_memory_label.pack(pady=5)

        console_frame = tk.Frame(memory_console_frame)
        console_frame.grid(row=0, column=1, sticky="nsew")

        self.console_box = scrolledtext.ScrolledText(
            console_frame, wrap=tk.WORD, width=116, height=5
        )
        self.console_box.grid(row=0, column=0, sticky="nsew")
        self.console_box.configure(state="disabled")

        input_frame = tk.Frame(main_frame, height=100)
        input_frame.grid(row=2, column=0, sticky="ew")
        input_frame.grid_rowconfigure(0, weight=1)
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = tk.Text(
            input_frame, width=60, height=5, wrap=tk.WORD
        )
        self.input_entry.grid(row=0, column=0, sticky="nsew")
        self.input_entry.bind("<Return>", self.send_message)
        self.input_entry.config(state="disabled")

        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            state="disabled",
        )
        self.send_button.grid(row=0, column=1, sticky="w")

        progress_frame = tk.Frame(main_frame, height=50)
        progress_frame.grid(row=3, column=0, sticky="ew")

        # Create the progress bar (self.progress_bar) here:
        self.progress_bar = ttk.Progressbar(
            progress_frame, mode="determinate"  # Set mode to determinate
        )
        self.progress_bar.pack(fill="x")

        self.model_path_frame = tk.Frame(main_frame, height=100)
        self.model_path_frame.grid(row=4, column=0, sticky="ew")
        self.model_path_frame.grid_rowconfigure(0, weight=1)
        self.model_path_frame.grid_columnconfigure(0, weight=1)
        self.model_path_frame.grid_columnconfigure(1, weight=10)
        self.model_path_frame.grid_columnconfigure(2, weight=1)

        # Label for the loaded model path
        self.model_path_label = tk.Label(self.model_path_frame, text="", wraplength=300)
        self.model_path_label.grid(row=0, column=1, sticky="w")

        tk.Label(self.model_path_frame, text="Load DEI").grid(
            row=0, column=0, sticky="w"
        )
        self.browse_button = tk.Button(
            self.model_path_frame, text="Browse", command=self.browse_model
        )
        self.browse_button.grid(row=0, column=2, sticky="w")

        # --- Parameter Adjustment Frame ---
        parameters_frame = tk.LabelFrame(main_frame, text="Generation Parameters", width=300, height=350)
        parameters_frame.grid(row=0, column=2, rowspan=3, sticky="nsew")  # Span 3 rows to align with other frames
        parameters_frame.grid_propagate(False)  # Prevent frame from resizing to fit children

        # --- Parameter Adjustment Widgets ---
        self.max_length_var = tk.IntVar(value=100)
        max_length_label = tk.Label(parameters_frame, text="Max Length:")
        max_length_label.grid(row=0, column=0, sticky="w")
        max_length_scale = tk.Scale(parameters_frame, from_=1, to=512, orient="horizontal", variable=self.max_length_var)
        max_length_scale.grid(row=0, column=1, sticky="ew")

        self.num_beams_var = tk.IntVar(value=1)
        num_beams_label = tk.Label(parameters_frame, text="Num Beams:")
        num_beams_label.grid(row=1, column=0, sticky="w")
        num_beams_scale = tk.Scale(parameters_frame, from_=1, to=10, orient="horizontal", variable=self.num_beams_var)
        num_beams_scale.grid(row=1, column=1, sticky="ew")

        self.no_repeat_ngram_size_var = tk.IntVar(value=0)
        no_repeat_ngram_size_label = tk.Label(parameters_frame, text="No Repeat N-gram Size:")
        no_repeat_ngram_size_label.grid(row=2, column=0, sticky="w")
        no_repeat_ngram_size_scale = tk.Scale(parameters_frame, from_=0, to=5, orient="horizontal", variable=self.no_repeat_ngram_size_var)
        no_repeat_ngram_size_scale.grid(row=2, column=1, sticky="ew")

        self.temperature_var = tk.DoubleVar(value=1.0)
        temperature_label = tk.Label(parameters_frame, text="Temperature:")
        temperature_label.grid(row=3, column=0, sticky="w")
        temperature_scale = tk.Scale(parameters_frame, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", variable=self.temperature_var)
        temperature_scale.grid(row=3, column=1, sticky="ew")

        self.top_k_var = tk.IntVar(value=50)
        top_k_label = tk.Label(parameters_frame, text="Top K:")
        top_k_label.grid(row=4, column=0, sticky="w")
        top_k_scale = tk.Scale(parameters_frame, from_=1, to=100, orient="horizontal", variable=self.top_k_var)
        top_k_scale.grid(row=4, column=1, sticky="ew")

        self.top_p_var = tk.DoubleVar(value=1.0)
        top_p_label = tk.Label(parameters_frame, text="Top P:")
        top_p_label.grid(row=5, column=0, sticky="w")
        top_p_scale = tk.Scale(parameters_frame, from_=0.0, to=1.0, resolution=0.1, orient="horizontal", variable=self.top_p_var)
        top_p_scale.grid(row=5, column=1, sticky="ew")

        self.do_sample_var = tk.BooleanVar(value=False)
        do_sample_checkbutton = tk.Checkbutton(parameters_frame, text="Do Sample", variable=self.do_sample_var)
        do_sample_checkbutton.grid(row=6, column=0, columnspan=2, sticky="w")

        # Attempt to load a trained model after progress bar is created
        self.load_model()

        if self.model and self.tokenizer:
            self.model_name = self.model.__class__.__name__
            self.dei_name = "DEI"
            self.input_entry.config(state="normal")
            self.send_button.config(state="normal")

    def browse_model(self):
        directory = filedialog.askdirectory()
        if directory:
            self.load_model(directory)
            print(f"DEI: {os.path.abspath(self.model_dir)}")

    def load_model(self, model_dir=None):
        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = find_first_valid_model_dir(os.path.join(".", "models", "educated-models"))

        if self.model_dir:
            # self.progress_bar.start()  <-- Remove this line
            if self.model_handler.load_model(self.model_dir):
                self.model = self.model_handler.model
                self.tokenizer = self.model_handler.tokenizer
                self.model_name = self.model_dir.split(os.path.sep)[-1]
                self.input_entry.config(state="normal")
                self.send_button.config(state="normal")
                self.model_path_label.config(text=f"Model: {os.path.abspath(self.model_dir)}")
                print(f"DEI: {os.path.abspath(self.model_dir)}")

                # Set the progress bar to full (100%) after loading
                self.progress_bar['value'] = 100  # <-- Add this line

            else:
                messagebox.showerror(
                    "Error", f"Failed to load model from {self.model_dir}"
                )

    def update_memory_labels(self, physical_memory, swap_memory, physical_label, swap_label):
        physical_label.config(text="Physical Memory: {:.1f} GB / {:.1f} GB".format(physical_memory.used / (1024**3), physical_memory.total / (1024**3),))
        swap_label.config(text="Swap Memory: {:.1f} GB / {:.1f} GB".format(swap_memory.used / (1024**3),swap_memory.total / (1024**3),))

    def send_message(self, event=None):
        prompt = self.input_entry.get("1.0", tk.END).strip()

        if prompt and self.model:
            self.dialogue_box.config(state="normal")
            self.dialogue_box.insert(tk.END, f"User: {prompt}\n\n")  # Added "User:" label
            self.dialogue_box.config(state="disabled")
            self.dialogue_box.see(tk.END)

            self.input_entry.delete("1.0", tk.END)  # Clear the input box

            self.progress_bar.start()

            try:
                input_ids = self.tokenizer.encode(
                    prompt, return_tensors="pt"
                ).to(get_device())

                # Use parameter values from the sliders/checkbutton
                output_ids = self.model.generate(
                    input_ids,
                    max_length=self.max_length_var.get(),
                    num_beams=self.num_beams_var.get(),
                    no_repeat_ngram_size=self.no_repeat_ngram_size_var.get(),
                    temperature=self.temperature_var.get(),
                    top_k=self.top_k_var.get(),
                    top_p=self.top_p_var.get(),
                    do_sample=self.do_sample_var.get()
                )

                response = self.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )

                self.dialogue_box.config(state="normal")
                self.dialogue_box.insert(tk.END, f"{self.dei_name}: {response}\n\n")  # Added model_name label
                self.dialogue_box.config(state="disabled")
                self.dialogue_box.see(tk.END)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                self.progress_bar.stop()
        else:
            messagebox.showwarning(
                "Warning", "Please enter a message or load a model."
            )
