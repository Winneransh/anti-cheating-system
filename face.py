#pip install insightface
#pip install onnxruntime
import insightface
import cv2
import numpy as np
import base64

def verify_with_insightface(image2_base64):
    try:
        # Initialize model
        app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        image1_base64=""# Replace with your first image base64 string
        # Decode base64 images
        img1_data = base64.b64decode(image1_base64)
        img2_data = base64.b64decode(image2_base64)
        
        # Convert to numpy arrays
        img1_array = np.frombuffer(img1_data, np.uint8)
        img2_array = np.frombuffer(img2_data, np.uint8)
        
        # Decode images using OpenCV
        img1 = cv2.imdecode(img1_array, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return False
        
        # Get face embeddings
        faces1 = app.get(img1)
        faces2 = app.get(img2)
        
        if len(faces1) == 0 or len(faces2) == 0:
            return False
        
        # Get embeddings
        emb1 = faces1[0].embedding
        emb2 = faces2[0].embedding
        
        # Calculate similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Return True if same person (similarity > 0.4)
        return similarity > 0.4
        
    except Exception as e:
        return False

# Usage example
# Convert your images to base64 first, then call:
# result = verify_with_insightface(image1_base64_string, image2_base64_string)
# print(result)  # Will print True or False

# Helper function to convert file to base64 (for testing)
#def file_to_base64(file_path):
#    with open(file_path, "rb") as image_file:
#        return base64.b64encode(image_file.read()).decode('utf-8')

# Example usage with files converted to base64:

img2_b64 = "your_base64_encoded_image_string_here"  # Replace with your second image base64 string
result = verify_with_insightface(img2_b64)
print(result)