import streamlit as st
from transformers import SamModel, SamProcessor
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt

# Load the processor and model from Hugging Face
model_id = "facebook/sam-vit-large"
processor = SamProcessor.from_pretrained(model_id)
model = SamModel.from_pretrained(model_id)

def load_image(image_file):
    """Load an image from a file-like object (uploaded by user)."""
    try:
        return Image.open(image_file).convert("RGB")
    except (UnidentifiedImageError) as e:
        st.error(f"Error loading image: {e}")
        return None

def segment_image(image):
    """Segment the image and return the masks."""
    # Prepare the input for the SAM model
    inputs = processor(images=image, return_tensors="pt")

    # Perform segmentation
    outputs = model(**inputs)
    masks = outputs.pred_masks.cpu().detach().numpy()

    return masks, image

def process_masks(masks):
    """Process and prepare masks for visualization."""
    # Select the first mask from the batch (if there's only one object)
    if masks.shape[0] == 1:
        mask = masks[0, 0, :, :]  # Select the first mask
    else:
        # Combine masks from multiple objects (sum of all masks)
        mask = np.max(masks[0], axis=0)  # Max across the channels (objects)

    # Ensure that the mask is 2D (height, width)
    if mask.ndim > 2:
        mask = mask[0, :, :]  # Reduce the dimensions if necessary

    return mask

def visualize_masks(image, masks):
    """Visualize the segmentation masks on the image."""
    if masks is None or image is None:
        st.error("No segmentation data available.")
        return

    # Process the mask for visualization
    processed_mask = process_masks(masks)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Normalize mask values for better visualization
    processed_mask = processed_mask / processed_mask.max()  # Normalize to [0, 1]

    # Display the mask overlaid on the original image
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(processed_mask, cmap="jet", alpha=0.5)
    ax.axis('off')
    st.pyplot(fig)

# Streamlit UI
st.title("Image Segmentation with SAM Model")

st.write("Upload an image to segment it using SAM!")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Load the image
    image = load_image(uploaded_file)

    if image:
        st.write("Segmenting the image...")
        masks, image = segment_image(image)

        if masks is not None:
            st.write(f"Number of objects segmented: {masks.shape[0]}")
            visualize_masks(image, masks)
