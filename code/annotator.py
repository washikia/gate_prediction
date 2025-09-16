import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import json
import os

class Annotator:
    def __init__(self, root, image_paths, save_file="annotations.json"):
        self.root = root
        self.image_paths = image_paths
        self.save_file = save_file
        self.index = 0
        self.points = {}  # filename -> list of (x,y)

        # Load existing annotations if file exists
        if os.path.exists(save_file):
            with open(save_file, "r") as f:
                self.points = json.load(f)

        # Canvas
        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill="both", expand=True)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="Prev", command=self.prev_image).pack(side="left")
        tk.Button(btn_frame, text="Next", command=self.next_image).pack(side="left")
        tk.Button(btn_frame, text="Save", command=self.save).pack(side="left")

        self.load_image()
        self.canvas.bind("<Button-1>", self.on_click)

    def load_image(self):
        path = self.image_paths[self.index]
        self.img = Image.open(path)
        self.tk_img = ImageTk.PhotoImage(self.img)
        self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # Draw saved points if exist
        filename = os.path.basename(path)
        if filename in self.points:
            for (x,y) in self.points[filename]:
                self.canvas.create_oval(x-3,y-3,x+3,y+3,fill="red")

    def on_click(self, event):
        x, y = event.x, event.y
        filename = os.path.basename(self.image_paths[self.index])
        if filename not in self.points:
            self.points[filename] = []
        self.points[filename].append((x,y))
        self.canvas.create_oval(x-3,y-3,x+3,y+3,fill="red")

    def save(self):
        with open(self.save_file, "w") as f:
            json.dump(self.points, f, indent=2)
        print("Saved annotations!")

    def next_image(self):
        if self.index < len(self.image_paths)-1:
            self.index += 1
            self.canvas.delete("all")
            self.load_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.canvas.delete("all")
            self.load_image()

# Run
if __name__ == "__main__":
    import glob
    image_folder = filedialog.askdirectory(title="Select image folder")
    image_paths = glob.glob(os.path.join(image_folder, "*.png")) + glob.glob(os.path.join(image_folder, "*.jpg"))

    root = tk.Tk()
    app = Annotator(root, image_paths)
    root.mainloop()
