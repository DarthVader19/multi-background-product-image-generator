import os
from PIL import Image
import numpy as np
from deeplabcut import DeepLabCut

# Set paths for images and output directory
product_dir = "images/products"
background_dir = "images/backgrounds"
output_dir = "output"
model_dir = "models"  # Downloaded segmentation model will be saved here

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Download pre-trained DeepLabCut model (modify URL if needed)
model = DeepLabCut(model_name="deeplabv2_resnet101_imagenet", download_dir=model_dir)

def generate_product_images(product_path):
  """
  Generates images with the product placed on various backgrounds.

  Args:
      product_path (str): Path to the product image.
  """
  # Load product image
  product_img = Image.open(product_path)
  product_img = np.array(product_img)

  # Perform image segmentation
  mask = model.predict(product_img)
  mask = mask.argmax(axis=2)  # Convert to single-channel mask

  # Extract product from original image using mask
  product = product_img[mask == 1]

  # Loop through background images
  for background_path in os.listdir(background_dir):
    background_img = Image.open(os.path.join(background_dir, background_path))
    background_img = np.array(background_img)

    # Resize product to match background dimensions
    product = Image.fromarray(product).resize(background_img.shape[:2], resample=Image.ANTIALIAS)
    product = np.array(product)

    # Inpaint any minor segmentation imperfections (optional)
    # inpainted_product = inpaint_image(product)  # Replace with your inpainting function

    # Place product on background
    background_img[mask == 0] = product

    # Optional: Apply style transfer for further enhancement (complex, not included here)
    # enhanced_image = style_transfer(background_img, product_img)  # Replace with your style transfer function

    # Save generated image
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(product_path))[0]}_{background_path}")
    Image.fromarray(background_img).save(output_path)

# Process all product images in the directory
for product_file in os.listdir(product_dir):
  product_path = os.path.join(product_dir, product_file)
  generate_product_images(product_path)

print("Product images with various backgrounds generated successfully!")
