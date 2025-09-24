import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Tabs Example")
root.geometry("400x300")

# Notebook (tab container)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# --- Tab 1: Hello App ---
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Hello App")

# Hello App content
name_var = tk.StringVar()

ttk.Label(tab1, text="Enter your name:").pack(pady=10)
entry = ttk.Entry(tab1, textvariable=name_var)
entry.pack(pady=5)

def say_hello():
    label.config(text=f"Hello, {name_var.get()}!")

btn = ttk.Button(tab1, text="Say Hello", command=say_hello)
btn.pack(pady=5)


def change_tab(current_tab):
    if current_tab == 1:
        notebook.select(tab2)
    else:
        notebook.select(tab1)


btn = ttk.Button(tab1, text="Go to next tab", command=lambda: change_tab(1))
btn.pack(pady=5)

label = ttk.Label(tab1, text="")
label.pack(pady=10)

# --- Tab 2: Survey Form ---
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Survey Form")

ttk.Label(tab2, text="Name:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
name_entry = ttk.Entry(tab2)
name_entry.grid(row=0, column=1, pady=5)

ttk.Label(tab2, text="Gender:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
gender = tk.StringVar()
ttk.Radiobutton(tab2, text="Male", variable=gender, value="Male").grid(row=1, column=1, sticky="w")
ttk.Radiobutton(tab2, text="Female", variable=gender, value="Female").grid(row=2, column=1, sticky="w")

ttk.Label(tab2, text="Favorite Color:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
color_combo = ttk.Combobox(tab2, values=["Red", "Blue", "Green"])
color_combo.grid(row=3, column=1, pady=5)
color_combo.current(0)

submit_btn = ttk.Button(tab2, text="Submit")
submit_btn.grid(row=4, column=0, columnspan=2, pady=10)

btn2 = ttk.Button(tab2, text="Go to previous tab", command=lambda: change_tab(0))
btn2.grid(row=5, column=0, columnspan=2, pady=5)

root.mainloop()
