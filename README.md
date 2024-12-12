# Image Caption Generator

This project implements an image caption generator using the BLIP (Bootstrapping Language-Image Pre-training) model from Salesforce. It can generate natural language descriptions for any input image.

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your image in the project directory
2. Modify the `image_path` variable in `image_caption_generator.py` to point to your image
3. Run the script:
```bash
python image_caption_generator.py
```

The script will:
- Load your image
- Generate a caption using the BLIP model
- Display the image with its caption using matplotlib

## Features

- Generates natural language captions for images
- Supports common image formats (JPEG, PNG, etc.)
- Displays the image alongside its generated caption
- Includes error handling for robust operation

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Pillow
- Matplotlib

## Model Details

This implementation uses the `Salesforce/blip-image-captioning-base` model, which is a state-of-the-art image captioning model that combines vision and language understanding.
