import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt

def generate_caption(image_path):
    """
    Generate a caption for the given image using the BLIP model.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        str: Generated caption describing the image
    """
    # Initialize the BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare the image for the model
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate caption
        outputs = model.generate(**inputs)
        
        # Decode the generated caption
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
        
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def display_image_and_caption(image_path):
    """
    Display the image and its generated caption.
    
    Args:
        image_path (str): Path to the input image file
    """
    # Generate caption
    caption = generate_caption(image_path)
    
    # Display image and caption
    image = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Caption: {caption}")
    plt.show()

if __name__ == "__main__":
    # Example usage with absolute path
    image_path = "D:/wifi/Generative AI/images/download.jpeg"
    display_image_and_caption(image_path)
