import tkinter as tk
from tkinter import ttk

root = tk.Tk()

display_var = tk.StringVar()

display = ttk.Entry(root, textvariable=display_var)
display.grid(row=0, column=0, columnspan=4, sticky="nsew")

buttons = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", ".", "=", "+"],
]



def on_button_click(button_text):
    current_text = display_var.get()
    if current_text == '0':
        display_var.set(button_text)
    else:
        display_var.set(current_text + button_text)

for r, row in enumerate(buttons, start=1):
    for c, btn in enumerate(row):
        button = ttk.Button(root, text=btn, command=lambda b=btn: on_button_click(b))
        button.grid(row=r, column=c, sticky="nsew")


for i in range(4):
    root.grid_columnconfigure(i, weight=1)

for j in range(5):
    root.grid_rowconfigure(j, weight=1)

root.mainloop()