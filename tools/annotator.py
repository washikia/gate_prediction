import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os, sys
import glob

class Annotator:
    def __init__(self, root, image_folder, save_file="annotations.json"):
        self.root = root
        self.root.title("Image Annotator")
        self.image_folder = image_folder
        self.save_file = save_file
        self.index = 0
        self.points = {}  # filename -> list of (x,y)
        self.years = []
        self.recently_deleted = []  # Stack to keep track of recently deleted points for undo

        # Make the label directory and load existing annotations if file exists
        self.save_file = os.path.join(os.path.dirname(self.image_folder), "labels", self.save_file)
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        if os.path.exists(self.save_file):
            with open(self.save_file, "r") as f:
                self.points = json.load(f)

        # Dropdown for year selection
        years = glob.glob(os.path.join(self.image_folder, "*"))
        years = [os.path.basename(y) for y in years if os.path.isdir(y)]
        self.years = years

        self.selected_year = tk.StringVar()
        top_frame = tk.Frame(root)
        top_frame.pack(anchor="nw", fill="x")

        self.year_combo = ttk.Combobox(top_frame, textvariable=self.selected_year, values=years)
        if years:
            self.year_combo.current(0)
        self.year_combo.pack(side="left", padx=5, pady=5)
        self.year_combo.bind("<<ComboboxSelected>>", self.on_year_change)

        self.undo_image = Image.open(self.resource_path("D:\\washik_personal\\projects\\gate_prediction\\tools\\asset\\undo.png"))
        self.undo_image = self.undo_image.resize((15, 15), Image.Resampling.LANCZOS)
        self.undo_image = ImageTk.PhotoImage(self.undo_image)

        self.redo_image = Image.open(self.resource_path("D:\\washik_personal\\projects\\gate_prediction\\tools\\asset\\redo.png"))
        self.redo_image = self.redo_image.resize((15, 15), Image.Resampling.LANCZOS)
        self.redo_image = ImageTk.PhotoImage(self.redo_image)

        undo_btn = tk.Button(top_frame, image=self.undo_image, command=self.delete_last_point)
        undo_btn.pack(side="right", padx=2)

        redo_btn = tk.Button(top_frame, image=self.redo_image, command=self.redo_last_point)
        redo_btn.pack(side="right", padx=2)

        # Progress label for x/y labeled images
        self.progress_label = tk.Label(top_frame, text="")
        self.progress_label.pack(side="left", padx=10)

        # Canvas
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill="both", expand=True)
        self.canvas_left = tk.Canvas(self.canvas_frame)
        self.canvas_left.pack(side="left", fill="both", expand=True)
        self.canvas_left.bind("<Button-1>", self.on_click)

        self.canvas_right = tk.Canvas(self.canvas_frame)
        self.canvas_right.pack(side="right", fill="both", expand=True)
        # self.canvas_right.bind("<Button-1>", self.on_click)

        self.root.bind("<Control-z>", self.delete_last_point)
        self.root.bind("<Control-s>", self.save)
        self.root.bind("<Control-S>", self.save)

        # Store image references
        self.tk_img = None
        self.tk_img_with_gate = None

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="Prev", command=self.prev_image).pack(side="left")
        tk.Button(btn_frame, text="Next", command=self.next_image).pack(side="left")
        tk.Button(btn_frame, text="Save", command=self.save).pack(side="left")
        tk.Button(btn_frame, text="Clear All", command=self.clear_all).pack(side="left")

        # Build image_paths for the first year
        self.image_paths = self.build_image_paths(self.selected_year.get())
        self.load_image()
        self.update_progress()  # Show initial progress

    def build_image_paths(self, selected_year):
        years = [selected_year] if selected_year else self.years.copy()
        image_paths = []
        for year in years:
            year_images = glob.glob("*.png", root_dir=os.path.join(self.image_folder, year, "without_gate"))
            for img in year_images:
                image_paths.append(os.path.join(self.image_folder, year, "without_gate", img))
        return image_paths

    def on_year_change(self, event=None):
        self.index = 0
        self.image_paths = self.build_image_paths(self.selected_year.get())
        self.load_image()

    def load_image(self):
        if not self.image_paths:
            print("No images found.")
            return
        path = self.image_paths[self.index]
        img = Image.open(path)
        # img = img.resize((400, 180), Image.Resampling.LANCZOS)
        # print("Opened image:", path)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas_left.config(width=self.tk_img.width(), height=self.tk_img.height())
        self.canvas_left.delete("all")
        self.canvas_left.create_image(0, 0, anchor="nw", image=self.tk_img)

        # with gate image
        path_with_gate = path.replace("without_gate", "with_gate")
        img_with_gate = Image.open(path_with_gate)
        # img_with_gate = img_with_gate.resize((400, 180), Image.Resampling.LANCZOS)
        self.tk_img_with_gate = ImageTk.PhotoImage(img_with_gate)
        self.canvas_right.config(width=self.tk_img_with_gate.width(), height=self.tk_img_with_gate.height())
        self.canvas_right.delete("all")
        self.canvas_right.create_image(0, 0, anchor="nw", image=self.tk_img_with_gate)

        # Draw saved points if exist
        filename = os.path.basename(path)
        if filename in self.points:
            for (x, y) in self.points[filename]:
                self.canvas_left.create_oval(x-3, y-3, x+3, y+3, fill="red")
                # self.canvas_right.create_oval(x-3, y-3, x+3, y+3, fill="red")

    def on_click(self, event):
        x, y = event.x, event.y
        filename = os.path.basename(self.image_paths[self.index])
        if filename not in self.points:
            self.points[filename] = []
        self.points[filename].append((x,y))
        self.canvas_left.create_oval(x-3,y-3,x+3,y+3,fill="red")
        # self.canvas_right.create_oval(x-3,y-3,x+3,y+3,fill="red")

    def save(self, event=None):
        print(self.save_file)
        with open(self.save_file, "w") as f:
            json.dump(self.points, f, indent=2)
        self.recently_deleted.clear()  # Clear redo stack on save
        print("Saved annotations!")
        self.update_progress()

    def next_image(self):
        if self.index < len(self.image_paths)-1:
            self.index += 1
            self.canvas_left.delete("all")
            # self.canvas_right.delete("all")
            self.recently_deleted.clear()  # Clear redo stack on image change
            self.load_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.canvas_left.delete("all")
            # self.canvas_right.delete("all")
            self.recently_deleted.clear()  # Clear redo stack on image change
            self.load_image()
    
    def clear_all(self):
        response = tk.messagebox.askyesno("Confirm", "Are you sure you want to clear all points for this image?")
        if not response:
            return

        image_name = os.path.basename(self.image_paths[self.index])
        if image_name in self.points:
            del self.points[image_name]
        self.canvas_left.delete("all")
        # self.canvas_right.delete("all")
        self.load_image()
        with open(self.save_file, "w") as f:
            json.dump(self.points, f, indent=2)
        print("Cleared all points for", image_name)
        self.update_progress()

    def delete_last_point(self, event=None):
        filename = os.path.basename(self.image_paths[self.index])
        if filename in self.points and self.points[filename]:
            last_point = self.points[filename].pop()
            self.recently_deleted.append(last_point)
            self.load_image()
            self.update_progress()

    def redo_last_point(self):
        if self.recently_deleted:
            last_point = self.recently_deleted.pop()
            filename = os.path.basename(self.image_paths[self.index])
            if filename not in self.points:
                self.points[filename] = []
            self.points[filename].append(last_point)
            self.load_image()
            self.update_progress()

    def update_progress(self):
        all_filenames = [os.path.basename(p) for p in self.image_paths]
        labeled_count = sum(1 for fname in all_filenames if fname in self.points and self.points[fname])
        total_count = len(all_filenames)
        self.progress_label.config(text=f"Labeled: {labeled_count} / {total_count}")
    
    def resource_path(self, relative_path):
        if hasattr(sys, 'MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)


# Run
if __name__ == "__main__":
    import glob
    image_folder = filedialog.askdirectory(title="Select image folder")

    root = tk.Tk()
    app = Annotator(root, image_folder)
    root.mainloop()
