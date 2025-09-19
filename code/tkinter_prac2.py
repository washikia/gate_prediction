import tkinter as tk
from tkinter import filedialog, messagebox

root = tk.Tk()
root.title("File Dialog")
root.geometry("600x600")



def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png, *.jpg, *.jpeg"), ("All Files", "*.*")])
    if file_path:
        label.config(text=file_path)
        image = tk.PhotoImage(file=file_path)
        image_label = tk.Label(root, image=image)
        image_label.image = image
        image_label.pack(pady=20)
    else:
        messagebox.showwarning("No Selection", "No file selected")



def save_file():
    save_path = filedialog.asksaveasfilename(
        title="Save a file",
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if save_path:
        label.config(text=f"Save as: {save_path}")
        # Optionally, write something to the file
        with open(save_path, "w") as f:
            f.write("Hello, this is a test file!")
    else:
        messagebox.showinfo("Cancelled", "Save operation was cancelled.")



button = tk.Button(root, text="Open File", command=open_file)
button.pack(pady=20)

btn_save = tk.Button(root, text="Save File", command=save_file)
btn_save.pack(pady=10)

label = tk.Label(root, text="No file selected")
label.pack(pady=20)


canvas = tk.Canvas(root, width=400, height=300, bg="white")
canvas.pack()

def draw_circle(event):
    x, y = event.x, event.y
    radius = 10
    canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="red")

canvas.bind("<Button-1>", draw_circle)


def on_click(event):
    global selected, start_x, start_y
    # Find closest shape to mouse click
    shape = canvas.find_closest(event.x, event.y)
    selected = shape[0]  # item ID
    start_x, start_y = event.x, event.y


def on_drag(event):
    global selected, start_x, start_y
    if selected:
        dx, dy = event.x - start_x, event.y - start_y
        canvas.move(selected, dx, dy)
        start_x, start_y = event.x, event.y


def on_release(event):
    global selected
    selected = None



canvas.create_rectangle(50, 50, 120, 120, fill="red")
canvas.create_oval(200, 100, 250, 150, fill="blue")
canvas.create_rectangle(300, 200, 400, 300, fill="green")


# Bind mouse events
canvas.bind("<Button-1>", on_click)
canvas.bind("<B1-Motion>", on_drag)
canvas.bind("<ButtonRelease-1>", on_release)


selected = None
start_x, start_y = 0, 0

root.mainloop()