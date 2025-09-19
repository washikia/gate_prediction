import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

root = tk.Tk()

title_label = tk.Label(root, text="Survey Form")
title_label.pack(pady=10)

form_frame = tk.Frame(root, background="lightgray")
form_frame.pack(padx=10, pady=10, fill="both", expand=True)

name = tk.StringVar()
name_label = tk.Label(form_frame, text="Name:")
name_label.pack(pady=5)

name_entry = tk.Entry(form_frame, textvariable=name)
name_entry.pack(pady=5)

gender = tk.StringVar(value="None")
g1 = tk.Radiobutton(form_frame, text="Male", variable=gender, value="Male")
g2 = tk.Radiobutton(form_frame, text="Female", variable=gender, value="Female")
g3 = tk.Radiobutton(form_frame, text="Other", variable=gender, value="Other")

g1.pack(pady=5)
g2.pack(pady=5)
g3.pack(pady=5)

combo = ttk.Combobox(form_frame, values=["Red", "Green", "Blue"])
combo.current(0)
combo.pack(pady=5)

var = tk.BooleanVar()
check = tk.Checkbutton(form_frame, text="Subscribe to newsletter", variable=var)
check.pack(pady=5)

age = tk.IntVar()
spin = tk.Spinbox(form_frame, from_=0, to=100, textvariable=age)
spin.pack(pady=5)

def submit_form():
    print("Name:", name.get())
    print("Gender:", gender.get())
    print("Favorite Color:", combo.get())
    print("Subscribed:", var.get())
    print("Age:", age.get())

button = tk.Button(form_frame, text="Submit", command=submit_form)
button.pack(pady=10)


menu_bar = tk.Menu(root)
root.config(menu=menu_bar)


def show_about():
    messagebox.showinfo("About", "MyApp v1.0\nCreated with Tkinter")


file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_showinfo("About", )

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Help", menu=help_menu)

file_menu.add_command(label="Exit", command=root.quit)
help_menu.add_command(label="About", command=lambda: print("This is a survey form application."))

toolbar = tk.Frame(root, bd=1, relief="raised")
btn_new = tk.Button(toolbar, text="New")
btn_open = tk.Button(toolbar, text="Open")

btn_new.pack(side="left", padx=2, pady=2)
btn_open.pack(side="left", padx=2, pady=2)

toolbar.pack(side="top", fill="x")

root.mainloop()