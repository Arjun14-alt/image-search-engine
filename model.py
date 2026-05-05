from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def safe_tensor(x):
    # 🔥 handles both tensor and weird outputs
    if hasattr(x, "cpu"):
        return x.cpu().numpy().astype("float32")
    elif hasattr(x, "last_hidden_state"):
        return x.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
    else:
        raise ValueError("Unexpected output type")


def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    return safe_tensor(features)


def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    return safe_tensor(features)