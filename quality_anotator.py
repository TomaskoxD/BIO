import cv2
import pandas as pd
import tkinter as tk
from tkinter import Text, END
from PIL import Image, ImageTk
import os
import glob

##############################################ZMENIT CESTU K SUBOROM##############################################
list_file = 'data/train_mine.csv'
path = 'images/train/'
##############################################ZMENIT CESTU K SUBOROM##############################################

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
    else:
        close_application(None)

def show_next_image():
    img_path = df['image'][current_index]
    img_path = img_path.replace('.jpeg', '.png')
    img = cv2.imread(path + img_path)

    img = cv2.resize(img, (900, 900))  # Adjust the image size
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img
    label2.configure(text=df['quality'][current_index])
    label3.configure(text=img_path)
    log_text.insert('1.0', f"Showing image {current_index + 1}\n")

def go_back():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_next_image()  # Show the previous image
        log_text.insert('1.0', f"Going back to image {current_index + 1}\n")

if not os.path.isfile(list_file):

    files = glob.glob(path + '*')
    image_names = [file.split('/')[-1] for file in files]
    # change from .png to .jpeg
    image_names = [file.split('.')[0] + '.jpeg' for file in image_names]
    df = pd.DataFrame(columns=['', 'image', 'quality', 'DR_grade'])
    df['image'] = image_names
    df['quality'] = 0
    df['DR_grade'] = 0
    df[''] =  df.index
    df.to_csv(list_file, index=False)
else:
    
    df = pd.read_csv(list_file)

root = tk.Tk()
root.title("Image Quality Rating Tool")
root.attributes('-fullscreen', True)
root.configure(background='black')

label = tk.Label(root)
label.pack()

label2 = tk.Label(root)
label2.pack()

label3 = tk.Label(root)
label3.pack()

# Buttons for rating image quality
btn_0 = tk.Button(root, text="Rate 0 - Reject", command=lambda: rate_image(0))
btn_0.pack()

btn_1 = tk.Button(root, text="Rate 1 - Usable", command=lambda: rate_image(1))
btn_1.pack()

btn_2 = tk.Button(root, text="Rate 2 - Good", command=lambda: rate_image(2))
btn_2.pack()

# Back button to go to the previous image
back_btn = tk.Button(root, text="Back", command=go_back)
back_btn.pack()

# Text widget for logging
log_text = Text(root, height=20, width=50)
log_text.pack()

# Initialize the log_text widget first
show_next_image()

root.bind('<Escape>', close_application)
root.mainloop()

# Display ratings at the end (you can save them to a file or use them as needed)
log_text.insert('1.0', "Image Ratings:\n")
for rating in ratings:
    log_text.insert('1.0', f"Image: {rating['image']}, Rating: {rating['rating']}\n")

log_text.config(state=tk.DISABLED)  # Disable text editing in the log widget
