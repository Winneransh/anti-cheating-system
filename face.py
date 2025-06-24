#pip install insightface
#pip install onnxruntime
import insightface
import cv2
import numpy as np

def verify_with_insightface(image2_path):
    try:
        # Initialize model
        app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        image1_path="WIN_20250622_02_22_25_Pro.jpg"
        
        # Load images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
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

# Usage
result = verify_with_insightface("WIN_20250622_21_03_45_Pro.jpg")
print(result)  # Will print True or False