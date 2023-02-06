import tensorflow as tf
import os
from PIL import Image, ImageDraw, ImageFont

# ------------ #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------ #
os.chdir(os.path.dirname(__file__))
model_path = r"../model/cats_vs_rabbits.h5"
model = tf.keras.models.load_model(model_path)
# ------------ #

def predict_image(img_path, size=(224, 224)):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size) / 255.

    y_pred = model.predict(tf.expand_dims(img, 0), verbose=0)

    return y_pred

def show_image(img_path, text):
    img = Image.open(img_path)
    w, h = img.size
    org = (int(w*0.04), int(h*0.03))
    font_size = max(30, int(((0.04*w) + (0.04*h)) / 2))

    font = ImageFont.truetype("C:\Windows\Fonts\Verdana.ttf", font_size)
    draw = ImageDraw.Draw(img)
    draw.multiline_text(org, text, font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(50, 50, 50))
    img.show()

def main(img_path):
    y_pred = predict_image(img_path)[0][0]

    if (y_pred >= 0.5):
        text = f"Rabbit: {y_pred:.2f}"
    else:
        text = f"Cat: {(1-y_pred):.2f}"
    
    show_image(img_path, text)

# ============================= #
while True:
    img_path = input("Enter image path: ").strip('"')

    main(img_path)

# ============================= #