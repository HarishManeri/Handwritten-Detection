# Handwritten-Detection
Cothon Internship
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class HandwritingRecognizer:
    def __init__(self):
        self.model = self.build_model()
        
    def build_model(self):
        """Build and return the CNN model"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def train_model(self, train_images, train_labels, epochs=10):
        """Train the model with provided data"""
        # Normalize pixel values
        train_images = train_images / 255.0
        
        # Reshape images for CNN
        train_images = train_images.reshape((-1, 28, 28, 1))
        
        # Train the model
        history = self.model.fit(train_images, train_labels, epochs=epochs,
                               validation_split=0.2)
        return history

    def preprocess_image(self, image_path):
        """Preprocess image for recognition"""
        # Read image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove noise
        denoised = cv2.medianBlur(binary, 3)
        
        return denoised, image

    def segment_lines(self, image):
        """Segment the image into lines of text"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes for each contour
        boxes = [cv2.boundingRect(c) for c in contours]
        
        # Sort boxes by y-coordinate (top to bottom)
        boxes = sorted(boxes, key=lambda x: x[1])
        
        return boxes

    def recognize_text(self, image, boxes):
        """Recognize text in the image"""
        recognized_text = []
        
        for box in boxes:
            x, y, w, h = box
            roi = image[y:y+h, x:x+w]
            
            # Convert to PIL Image
            pil_image = Image.fromarray(roi)
            
            # Use Tesseract for text recognition
            text = pytesseract.image_to_string(pil_image, config='--psm 6')
            recognized_text.append(text.strip())
        
        return recognized_text

    def enhance_image(self, image):
        """Enhance image quality"""
        # Apply adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply dilation
        kernel = np.ones((1, 1), np.uint8)
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)
        
        return enhanced

    def process_document(self, image_path):
        """Process a complete document"""
        # Preprocess image
        processed_image, original_image = self.preprocess_image(image_path)
        
        # Enhance image
        enhanced_image = self.enhance_image(processed_image)
        
        # Segment into lines
        boxes = self.segment_lines(enhanced_image)
        
        # Recognize text
        recognized_text = self.recognize_text(enhanced_image, boxes)
        
        return recognized_text, original_image, boxes

    def display_results(self, image, boxes, recognized_text):
        """Display recognition results"""
        result_image = image.copy()
        
        for box, text in zip(boxes, recognized_text):
            x, y, w, h = box
            # Draw rectangle around text
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Put recognized text above the rectangle
            cv2.putText(result_image, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display result
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Example usage
def main():
    # Initialize recognizer
    recognizer = HandwritingRecognizer()
    
    # Process a sample image
    image_path = "path_to_your_handwritten_image.jpg"
    
    try:
        # Process the document
        recognized_text, original_image, boxes = recognizer.process_document(image_path)
        
        # Display results
        recognizer.display_results(original_image, boxes, recognized_text)
        
        # Print recognized text
        print("\nRecognized Text:")
        for i, text in enumerate(recognized_text, 1):
            print(f"Line {i}: {text}")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")

# Additional utility functions
def save_results(recognized_text, output_path):
    """Save recognized text to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in recognized_text:
            f.write(text + '\n')

def batch_process_directory(directory_path, recognizer):
    """Process all images in a directory"""
    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            recognized_text, _, _ = recognizer.process_document(image_path)
            results[filename] = recognized_text
    return results

# Custom preprocessing for specific types of documents
def preprocess_for_math(image):
    """Special preprocessing for mathematical expressions"""
    # Add specialized preprocessing for mathematical notation
    pass

def preprocess_for_drawings(image):
    """Special preprocessing for drawings/diagrams"""
    # Add specialized preprocessing for drawings
    pass

if __name__ == "__main__":
    main()
