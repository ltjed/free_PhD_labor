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
       with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            data = data[8:]  # As in your code

            root = tk.Tk()
            root.title("JSON Viewer")
            root.geometry("800x600")

            text_area = scrolledtext.ScrolledText(root, width=80, height=30, font=('Consolas', 12))
            text_area.pack(padx=10, pady=10, expand=True, fill='both')

            # Convert Python object to a JSON string with indentation
            # Use ensure_ascii=False so that non-ASCII characters stay in Unicode form
            json_str = json.dumps(data, indent=4, ensure_ascii=False)

            # If your JSON contains literal backslash-n (\\n) that needs to be turned
            # into an actual newline, you can keep the following replacement:
            json_str = json_str.replace("\\n", "\n")

            # Optionally add an extra blank line after each newline for readability
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