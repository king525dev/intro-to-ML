# gui.py

import tkinter as tk
from tkinter import scrolledtext, messagebox
from generate import generate_text

def generate():
    prompt = prompt_entry.get()
    if not prompt.strip():
        messagebox.showwarning("Missing Prompt", "Please enter a prompt.")
        return
    try:
        length = int(length_entry.get())
    except ValueError:
        messagebox.showerror("Invalid Input", "Length must be an integer.")
        return
    result = generate_text(prompt, length)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, result)

# Window setup
root = tk.Tk()
root.title("🧠✌️ VibeAI")
root.geometry("600x500")
root.configure(bg="#1e1e1e")

# Prompt Entry
tk.Label(root, text="Prompt:", bg="#1e1e1e", fg="white").pack(anchor="w", padx=10, pady=(10, 0))
prompt_entry = tk.Entry(root, width=60, font=("Courier", 12))
prompt_entry.pack(padx=10, pady=5)

# Length Entry
tk.Label(root, text="Length:", bg="#1e1e1e", fg="white").pack(anchor="w", padx=10)
length_entry = tk.Entry(root, width=10, font=("Courier", 12))
length_entry.insert(0, "200")
length_entry.pack(padx=10, pady=5)

# Generate Button
tk.Button(root, text="Generate", command=generate, bg="#444", fg="white", font=("Courier", 12)).pack(pady=10)

# Output Text Box
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier", 12), bg="#111", fg="white", height=15)
output_text.pack(padx=10, pady=10, fill="both", expand=True)

# Run the app
root.mainloop()