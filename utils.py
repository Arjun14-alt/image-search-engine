import os

def save_uploaded_file(uploaded_file):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(BASE_DIR)

    folder = os.path.join(project_root, "data", "images")

    # 🔥 safe folder creation
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path