
from deepface import DeepFace

def check_face(test_image):
    reference = "/content/WIN_20250622_21_03_45_Pro.jpg"
    
    result = DeepFace.verify(reference, test_image)
    
    if result['verified']:
        return "same"
    else:
        return "different"

# Usage
result = check_face("/content/your_test_image.jpg")
print(result)