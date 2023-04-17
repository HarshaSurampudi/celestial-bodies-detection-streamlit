import streamlit as st
import json
from PIL import Image
from fastai.vision.all import *

learn_inf = load_learner('celestial.pkl')


# Function to load details from JSON file
def load_details(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


# Function to display the details of the identified class
def display_details(details, class_name, prob):
    if class_name in details:
        image_path = details[class_name]["image"]
        st.image(image_path, caption=f"{details[class_name]['name']} Image", width=300)
        st.write("### ", details[class_name]["name"])
        # round to 2 decimal places with % suffix
        st.write("**Probability:** ", "{:.2f} %".format(prob*100))
        st.write("**Description:** ", details[class_name]["description"])
    else:
        st.write("Class not found in the details file")


# Load the details from the JSON file
celestial_details = load_details("celestial_details.json")


# Title
st.title("Celestial Body Image Classifier")

# Instructions
st.write("Upload an image of a celestial body (planets, moons, asteroids, etc.) and get its classification details.")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "JPG", "JPEG"])

if uploaded_file:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)


    # Submit button
    if st.button("Classify"):
        # Call the image classification logic here
        # For demonstration purposes, we assume the class_name is returned from the classification logic
        pred, pred_idx, probs = learn_inf.predict(image)
        prob = probs[pred_idx]
        class_name = pred  # Replace this with the output of your classification logic

        # Display the details of the identified class
        display_details(celestial_details, class_name , prob)