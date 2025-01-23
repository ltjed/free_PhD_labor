import json
from pprint import pprint
from rich import print_json
import tkinter as tk
from tkinter import scrolledtext
import os.path as osp
import tkinter as tk
from tkinter import scrolledtext
import json

import tkinter as tk
from tkinter import scrolledtext
import json

def display_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            data = data[8:]
                        
            root = tk.Tk()
            root.title("JSON Viewer")
            root.geometry("800x600")

            text_area = scrolledtext.ScrolledText(root, width=80, height=30, font=('Consolas', 12))
            text_area.pack(padx=10, pady=10, expand=True, fill='both')

            # Convert Python dict to JSON string with indentation
            json_str = json.dumps(data, indent=4)

            # Convert any literal "\n" in the text to actual newlines
            # Only needed if the JSON text *literally* contains backslash-n (\\n)
            json_str = json_str.replace("\\n", "\n")

            # Add extra newline after each entry
            json_str = json_str.replace("\n", "\n\n")

            # Insert the text into the ScrolledText
            text_area.insert(tk.END, json_str)

            text_area.configure(state='disabled')
            root.mainloop()

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        print("Error: Invalid JSON format")

if __name__ == "__main__":
    # Replace with your JSON file path
    base_dir = osp.join("templates", "sae_variants")
    
    json_file_path = osp.join(base_dir, "ideas.json")
    display_json(json_file_path)