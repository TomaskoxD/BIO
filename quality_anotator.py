import cv2
import pandas as pd
import tkinter as tk
from tkinter import Text, END
from PIL import Image, ImageTk
import os
import glob

# Modify the file path and directory path as needed
list_file = 'data/test_mine.csv'
path = 'images/test/'
# resolution = (1900, 1060) # 1080p
resolution = (2540, 1420) # 2K
# resolution = (3820, 2140) # 4K


current_index = 0
ratings = []

def close_application(event):
    print("Closing application")
    root.quit()

def rate_image(rating):
    global current_index
    global ratings
    df.at[current_index, 'quality'] = rating
    ratings.append({'image': df['image'][current_index], 'rating': rating})
    current_index += 1

    if current_index < len(df):
        show_next_image()
        df.to_csv(list_file, index=False)
        log_text.insert('1.0', f"Saved ratings to test.csv\n")
        update_status_label()
    else:
        close_application(None)

def show_next_image():
    img_path = df['image'][current_index]
    img_path = img_path.replace('.jpeg', '.png')
    img = cv2.imread(os.path.join(path, img_path))

    img = cv2.resize(img, (900, 900))
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img
    label2.configure(text=df['quality'][current_index])
    label3.configure(text=img_path)
    log_text.insert('1.0', f"Showing image {current_index + 1}\n")
    update_status_label()

def go_back():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_next_image()
        log_text.insert('1.0', f"Going back to image {current_index + 1}\n")
        update_status_label()

def update_status_label():
    status_label.configure(text=f"Image {current_index + 1}/{len(df)}")

if not os.path.isfile(list_file):
    files = glob.glob(os.path.join(path, '*'))
    image_names = [os.path.basename(file) for file in files]
    image_names = [file.split('.')[0] + '.jpeg' for file in image_names]
    df = pd.DataFrame(columns=['', 'image', 'quality', 'DR_grade'])
    df['image'] = image_names
    df['quality'] = 0
    df['DR_grade'] = 0
    df[''] = df.index
    df.to_csv(list_file, index=False)
else:
    df = pd.read_csv(list_file)

root = tk.Tk()
root.title("Image Quality Rating Tool")
root.geometry(f"{resolution[0]}x{resolution[1]}+0+0")
root.configure(background='black')

label = tk.Label(root)
label.pack()

label2 = tk.Label(root)
label2.pack()

label3 = tk.Label(root)
label3.pack()

status_label = tk.Label(root, text="Image 1/1")  # Initialize with "Image 1/1"
status_label.pack()

btn_0 = tk.Button(root, text="Rate 0 - Reject", command=lambda: rate_image(0))
btn_0.pack()

btn_1 = tk.Button(root, text="Rate 1 - Usable", command=lambda: rate_image(1))
btn_1.pack()

btn_2 = tk.Button(root, text="Rate 2 - Good", command=lambda: rate_image(2))
btn_2.pack()

back_btn = tk.Button(root, text="Back", command=go_back)
back_btn.pack()

log_text = Text(root, height=20, width=50)
log_text.pack()

show_next_image()

root.bind('<Escape>', close_application)
root.mainloop()

log_text.insert('1.0', "Image Ratings:\n")
for rating in ratings:
    log_text.insert('1.0', f"Image: {rating['image']}, Rating: {rating['rating']}\n")

log_text.config(state=tk.DISABLED)
