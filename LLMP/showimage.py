import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, images, result):
        self.images = images  # List of image tuples
        self.result = result  # The result dictionary containing ground truth and other data
        self.converted_images = []  # This will hold the converted images after processing

        # Automatically process images and display them with ground truth upon instantiation
        self.process_images()
        self.showimage()
        self.show_ground_truth()

    # Function to extract image data (first element of the tuple)
    def extract_image_data(self, image_tuple):
        image_data = image_tuple[0]  # Extract the boolean array from the tuple
        return image_data

    # Function to convert boolean image data to binary grayscale and then RGB
    def convert_to_grayscale(self, image_tuple):
        image = self.extract_image_data(image_tuple)  # Extract image data (boolean array)
        
        size = image.shape[0]
        grayscale = np.zeros((size, size), dtype=np.uint8)
        grayscale[image == False] = 255  # Set False (background) to white (255)
        grayscale[image == True] = 0     # Set True (foreground) to black (0)
        
        pil_image = Image.fromarray(grayscale)  # Convert NumPy array to PIL image
        
        # Convert the grayscale image to RGB
        rgb_image = pil_image.convert("RGB")
        return rgb_image

    # Function to process and store the images in RGB format
    def process_images(self):
        self.converted_images = [self.convert_to_grayscale(image) for image in self.images]

    # Function to display images using matplotlib
    def showimage(self):
        if not self.converted_images:
            self.process_images()  # Ensure images are processed before displaying
        
        fig, axes = plt.subplots(1, len(self.converted_images), figsize=(15, 5))  # Create subplots
        if len(self.converted_images) == 1:
            axes = [axes]  # Wrap single axes object into a list for consistency
        
        for ax, img in zip(axes, self.converted_images):
            ax.imshow(img)  # Display the RGB image
            ax.axis('off')  # Hide the axis
        
        plt.show()

    # Function to display ground truth values
    def show_ground_truth(self):
        print("Ground Truth:", self.result['gt'])