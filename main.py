import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from app.gui import HomeWindow, CreationWindow, InteractionWindow, DownloadWindow
from app.bootstrap import check_system_memory, find_first_valid_model_dir, load_model_and_tokenizer, get_device
import queue
import sys
import os
import time
import logging
from threading import Thread, Lock  # Import Lock
import datetime
import subprocess

# ------------------------------
# Logging Setup
# ------------------------------

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------
# Console Output Redirection
# ------------------------------


class RedirectStdout:
    """Redirects standard output to a queue for display in a Tkinter Text widget."""

    def __init__(self, queue, window_name, redirect_stdout_dict): # Add redirect_stdout_dict here
        logging.debug("Initializing RedirectStdout for window: %s", window_name)
        self.queue = queue
        self.stdout = sys.stdout
        self.window_name = window_name
        self.redirect_stdout_dict = redirect_stdout_dict # Store the dictionary
        self.queue_lock = Lock()

    def write(self, text):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"[{timestamp}] {text}"

        logging.debug("Redirecting output: %s", formatted_text)

        # Get the current RedirectStdout instance from the dictionary
        current_redirect = self.redirect_stdout_dict.get(self.window_name)

        if current_redirect:
            with current_redirect.queue_lock:  # Acquire the lock before writing
                try:
                    current_redirect.queue.put_nowait(
                        (formatted_text, "console", self.window_name)
                    )
                except queue.Full:
                    logging.warning("Output queue full. Dropping message: %s", formatted_text)

    def flush(self):
        logging.debug("RedirectStdout flush called (doing nothing)")
        pass


# ----------------------
# Console Manager Class
# ----------------------


class ConsoleManager:
    """Manages console output, handling queue processing and GUI updates."""

    def __init__(self, master, output_queue):
        """
        Initializes the ConsoleManager.

        Args:
            master (MainApplication): A reference to the MainApplication instance.
            output_queue (queue.Queue): The queue to monitor for console output messages.
        """
        logging.debug("Initializing ConsoleManager")
        self.master = master  # Store a reference to MainApplication
        self.output_queue = output_queue

        # Start the thread to process the output queue.
        self.update_thread = Thread(target=self._update_output, daemon=True)
        self.update_thread.start()

    def _update_output(self):
        """Continuously processes the output queue and updates the console boxes."""
        logging.debug("ConsoleManager update thread started")
        while True:
            if not self.output_queue.empty():
                item, target, window_name = self.output_queue.get()

                logging.debug("Processing message: %s (target: %s, window: %s)", item, target, window_name)

                if callable(item):
                    try:
                        item()
                    except Exception as e:
                        logging.error("Error executing callable: %s", e)
                else:
                    if target == "console":
                        # Schedule GUI update using after()
                        self.master.master.after(10, self._update_console_box, window_name, item)

            time.sleep(0.1)

    def _update_console_box(self, window_name, item):
        """Updates the console box of the specified window."""
        try:
            for tab_id in self.master.notebook.tabs():
                tab_name = self.master.notebook.tab(tab_id, 'text')
                if window_name == tab_name and hasattr(self.master.notebook.nametowidget(tab_id), "console_box"):
                    console_box = self.master.notebook.nametowidget(tab_id).console_box
                    console_box.config(state="normal")
                    console_box.insert(tk.END, item)
                    console_box.see(tk.END)
                    console_box.config(state="disabled")
                    logging.debug("Console box updated for window: %s", window_name)
                    break
        except Exception as e:
            logging.error("Error updating console box: %s", e)


# ----------------------
# Main Application Window
# ----------------------


class MainApplication:
    """Represents the main application window and manages GUI interactions."""

    def __init__(self, master):
        logging.debug("Initializing MainApplication")
        self.master = master
        self.master.title("DEI Toolkit")
        self.master.geometry("1350x700")

        # Configure grid layout
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=0)
        self.master.grid_columnconfigure(0, weight=1)

        # Create queue for output messages
        self.output_queue = queue.Queue()

        # Create dictionary for window-specific redirection
        self.redirect_stdout_dict = {}

        # Redirect standard output to the queue for the initial Home window
        self.redirect_stdout_dict["Home"] = RedirectStdout(self.output_queue, "Home", self.redirect_stdout_dict)
        sys.stdout = self.redirect_stdout_dict["Home"]

        # Create necessary directories if they don't exist
        required_dirs = [
            "models",
            os.path.join("models", "stock-models"),
            os.path.join("models", "educated-models"),
            "courses",
        ]
        for dir_path in required_dirs:
            full_path = os.path.join(os.getcwd(), dir_path)
            if not os.path.exists(full_path):
                try:
                    os.makedirs(full_path)
                    print(f"Created directory: {full_path}")
                except OSError as e:
                    print(f"Error creating directory {full_path}: {e}")

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.master)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.create_navigation_bar()

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None

        # Show the home window
        self.show_root_window()
        ConsoleManager(self, self.output_queue)  # Pass 'self' to ConsoleManager

    def create_navigation_bar(self):
        """Creates the navigation bar with buttons for navigating between windows."""
        self.nav_bar = tk.Frame(self.master, width=1400, height=50)
        self.nav_bar.grid(row=1, column=0, sticky="ew")

        self.back_button = tk.Button(
            self.nav_bar, text="Back", command=self.show_root_window
        )
        self.back_button.pack(side=tk.LEFT, padx=5)

        self.create_button_nav = tk.Button(
            self.nav_bar, text="Create", command=self.show_creation_window
        )
        self.create_button_nav.pack(side=tk.LEFT, padx=5)

        self.interact_button_nav = tk.Button(
            self.nav_bar, text="Interact", command=self.show_interaction_window
        )
        self.interact_button_nav.pack(side=tk.LEFT, padx=5)

        self.download_button_nav = tk.Button(
            self.nav_bar, text="Download", command=self.show_download_window
        )
        self.download_button_nav.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(
            self.nav_bar, text="New Slate", command=self.reset_app
        )
        self.reset_button.pack(side=tk.RIGHT, padx=5)

    def show_root_window(self):
        """Displays the HomeWindow."""
        for i in range(self.notebook.index("end") - 1, -1, -1):
            self.notebook.forget(i)

        self.root_frame = tk.Frame(self.notebook, width=1400, height=950)
        self.notebook.add(self.root_frame, text="Home")

        self.root_frame.grid_rowconfigure(0, weight=1)
        self.root_frame.grid_rowconfigure(1, weight=1)
        self.root_frame.grid_columnconfigure(0, weight=1)

        # Make sure sys.stdout is redirected to the Home window's console
        sys.stdout = self.redirect_stdout_dict.get("Home", RedirectStdout(self.output_queue, "Home", self.redirect_stdout_dict))

        HomeWindow(
            self.root_frame, self, self.output_queue, self.notebook
        ).pack(expand=True, fill="both")
        self.output_queue.put(("Home", "console", "Home"))

    def show_creation_window(self):
        """Displays the CreationWindow."""
        output_queue = self.output_queue
        gui_update_queue = queue.Queue()

        # Create a window-specific RedirectStdout for the Creation window
        if "Creation" not in self.redirect_stdout_dict:
            self.redirect_stdout_dict["Creation"] = RedirectStdout(output_queue, "Creation", self.redirect_stdout_dict)

        # Set sys.stdout to redirect to the Creation window's console
        sys.stdout = self.redirect_stdout_dict["Creation"]

        if not hasattr(self, "creation_frame"):
            self.creation_frame = tk.Frame(self.notebook)
            self.notebook.add(self.creation_frame, text="Creation")

        if not hasattr(self, "creation_app"):
            self.creation_app = CreationWindow(  # Modified line
                self.creation_frame,
                output_queue=output_queue,
                gui_update_queue=gui_update_queue,
                notebook=self.notebook,
                redirect_stdout=self.redirect_stdout_dict["Creation"],
                main_app=self  # Added line: Pass the MainApplication instance
            )
            self.creation_app.update_output()

        self.notebook.select(self.creation_frame)
        self.output_queue.put(("Creation", "console", "Creation"))

        self.back_button.config(state="normal")
        self.interact_button_nav.config(state="normal")
        self.download_button_nav.config(state="normal")

    def show_interaction_window(self):
        """Displays the InteractionWindow."""
        if hasattr(self, "interaction_frame"):
            self.notebook.select(self.interaction_frame)
            return

        self.interaction_frame = tk.Frame(self.notebook)
        self.notebook.add(self.interaction_frame, text="Interaction")

        model_dir = find_first_valid_model_dir(os.path.join(".", "models", "educated-models"))
        if model_dir:
            try:
                self.model, self.tokenizer = load_model_and_tokenizer(
                    model_dir
                )
                self.model.to(get_device())
                self.model_name = model_dir.split(os.path.sep)[-1]
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to load model: {e}"
                )
                self.model = None
                self.tokenizer = None
        else:
            messagebox.showinfo(
                "Information",
                "No trained DEI model found. Please train a model first.",
            )

        # Create a window-specific RedirectStdout for the Interaction window
        if "Interaction" not in self.redirect_stdout_dict:
            self.redirect_stdout_dict["Interaction"] = RedirectStdout(
                self.output_queue, "Interaction", self.redirect_stdout_dict
            )

        # Set sys.stdout to redirect to the Interaction window's console
        sys.stdout = self.redirect_stdout_dict["Interaction"]

        self.interaction_window = InteractionWindow(
            self.interaction_frame,
            self,
            self.model,
            self.tokenizer,
            self.notebook,
        )
        self.interaction_window.pack(expand=True, fill="both")

        physical_memory, swap_memory = check_system_memory()
        self.update_memory_labels(
            physical_memory,
            swap_memory,
            self.interaction_window.physical_memory_label,
            self.interaction_window.swap_memory_label,
        )

        self.notebook.select(self.interaction_frame)
        self.output_queue.put(("Interaction", "console", "Interaction"))

        self.back_button.config(state="normal")
        self.create_button_nav.config(state="normal")
        self.download_button_nav.config(state="normal")

    def show_download_window(self):
        """Displays the DownloadWindow."""
        if not hasattr(self, "download_frame"):
            self.download_frame = tk.Frame(self.notebook)
            self.notebook.add(self.download_frame, text="Download")

        if not hasattr(self, "download_app"):
            # Redirect stdout for DownloadWindow
            if "Download" not in self.redirect_stdout_dict:
                self.redirect_stdout_dict["Download"] = RedirectStdout(
                    self.output_queue, "Download", self.redirect_stdout_dict
                )
            sys.stdout = self.redirect_stdout_dict["Download"]

            self.download_app = DownloadWindow(
                self.download_frame, self, self.output_queue, self.notebook
            )
            self.download_app.pack(expand=True, fill="both")

        self.notebook.select(self.download_frame)
        self.output_queue.put(("Download", "console", "Download"))

        self.back_button.config(state="normal")

    def reset_app(self):
        """Resets the application to its initial state."""
        if messagebox.askyesno(
            "Confirm Reset",
            "Are you sure you want to reset the application? This will close all windows and clear any data.",
        ):
            self.master.destroy()
            # Restart the application
            os.execl(sys.executable, sys.executable, *sys.argv)

    def update_memory_labels(
        self, physical_memory, swap_memory, physical_label, swap_label
    ):
        """Updates memory usage labels in the InteractionWindow."""
        physical_label.config(
            text="Physical Memory: {:.1f} GB / {:.1f} GB".format(
                physical_memory.used / (1024**3),
                physical_memory.total / (1024**3),
            )
        )
        swap_label.config(
            text="Swap Memory: {:.1f} GB / {:.1f} GB".format(
                swap_memory.used / (1024**3),
                swap_memory.total / (1024**3),
            )
        )


# -----------------------------------
# Main Function & GUI Initialization
# -----------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.main_app = app
    root.mainloop()
