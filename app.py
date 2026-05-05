import streamlit as st
from model import get_image_embedding, get_text_embedding
from search import add_embedding, search
from utils import save_uploaded_file
from PIL import Image

st.title("🖼️ Image Search Engine")

# Upload images
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        path = save_uploaded_file(file)
        embedding = get_image_embedding(path)
        add_embedding(embedding, path)

    st.success("Images uploaded and processed!")

# Search
query = st.text_input("Search for images")

if st.button("Search") and query:
    query_embedding = get_text_embedding(query)
    results = search(query_embedding)

    st.subheader("Results:")
    for img_path in results:
        image = Image.open(img_path)
        st.image(image, width=200)