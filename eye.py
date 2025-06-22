import cv2
import mediapipe as mp
import numpy as np
import math

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Iris landmarks (MediaPipe provides these with refine_landmarks=True)
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
    
    def get_eye_landmarks(self, landmarks, eye_indices):
        """Extract eye landmarks as numpy array"""
        return np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    
    def get_iris_center(self, landmarks, iris_indices):
        """Calculate iris center from iris landmarks"""
        iris_points = np.array([(landmarks[i].x, landmarks[i].y) for i in iris_indices])
        return np.mean(iris_points, axis=0)
    
    def get_eye_center(self, eye_points):
        """Calculate eye center from eye corner points"""
        return np.mean(eye_points, axis=0)
    
    def calculate_gaze_ratio(self, iris_center, eye_points):
        """Calculate horizontal and vertical gaze ratios"""
        eye_center = self.get_eye_center(eye_points)
        
        # Get eye dimensions
        eye_width = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])
        eye_height = np.max(eye_points[:, 1]) - np.min(eye_points[:, 1])
        
        # Calculate relative position of iris
        horizontal_ratio = (iris_center[0] - eye_center[0]) / eye_width if eye_width > 0 else 0
        vertical_ratio = (iris_center[1] - eye_center[1]) / eye_height if eye_height > 0 else 0
        
        return horizontal_ratio, vertical_ratio
    
    def determine_gaze_direction(self, left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio):
        """Determine gaze direction based on iris positions - returns single direction only"""
        # Average the ratios from both eyes
        avg_horizontal = (left_h_ratio + right_h_ratio) / 2
        avg_vertical = (left_v_ratio + right_v_ratio) / 2
        
        # Thresholds for gaze direction (you can adjust these)
        h_threshold = 0.01
        v_threshold = 0.1
        
        # Get absolute values to determine which direction is more dominant
        abs_horizontal = abs(avg_horizontal)
        abs_vertical = abs(avg_vertical)
        
        # If both are below threshold, looking forward
        if abs_horizontal < h_threshold and abs_vertical < v_threshold:
            return "FORWARD"
        
        # Determine which direction is more dominant
        if abs_horizontal > abs_vertical:
            # Horizontal movement is more dominant
            if avg_horizontal < -h_threshold:
                return "RIGHT"
            elif avg_horizontal > h_threshold:
                return "LEFT"
            else:
                return "FORWARD"
        else:
            # Vertical movement is more dominant
            if avg_vertical < -v_threshold:
                return "UP"
            elif avg_vertical > v_threshold:
                return "DOWN"
            else:
                return "FORWARD"
    
    def analyze_gaze(self, image_path):
        """Main function to analyze gaze direction from an image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, "No face detected in the image"
        
        # Get the first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get eye and iris landmarks
        left_eye_points = self.get_eye_landmarks(face_landmarks.landmark, self.LEFT_EYE)
        right_eye_points = self.get_eye_landmarks(face_landmarks.landmark, self.RIGHT_EYE)
        
        left_iris_center = self.get_iris_center(face_landmarks.landmark, self.LEFT_IRIS)
        right_iris_center = self.get_iris_center(face_landmarks.landmark, self.RIGHT_IRIS)
        
        # Calculate gaze ratios
        left_h_ratio, left_v_ratio = self.calculate_gaze_ratio(left_iris_center, left_eye_points)
        right_h_ratio, right_v_ratio = self.calculate_gaze_ratio(right_iris_center, right_eye_points)
        
        # Determine gaze direction
        gaze_direction = self.determine_gaze_direction(left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio)
        
        # Create result dictionary
        result = {
            'gaze_direction': gaze_direction,
            'left_eye_ratios': {'horizontal': left_h_ratio, 'vertical': left_v_ratio},
            'right_eye_ratios': {'horizontal': right_h_ratio, 'vertical': right_v_ratio},
            'average_ratios': {
                'horizontal': (left_h_ratio + right_h_ratio) / 2,
                'vertical': (left_v_ratio + right_v_ratio) / 2
            }
        }
        
        return result, "Success"
    
    def analyze_and_visualize(self, image_path, output_path=None):
        """Analyze gaze and optionally save visualization"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, "No face detected in the image"
        
        # Analyze gaze
        result, status = self.analyze_gaze(image_path)
        
        if result is None:
            return None, status
        
        # Draw landmarks on image for visualization
        annotated_image = image.copy()
        face_landmarks = results.multi_face_landmarks[0]
        
        h, w, _ = annotated_image.shape
        
        # Draw eye contours
        for eye_indices in [self.LEFT_EYE, self.RIGHT_EYE]:
            eye_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) 
                         for i in eye_indices]
            cv2.polylines(annotated_image, [np.array(eye_points)], True, (0, 255, 0), 1)
        
        # Draw iris centers
        for iris_indices in [self.LEFT_IRIS, self.RIGHT_IRIS]:
            iris_center = self.get_iris_center(face_landmarks.landmark, iris_indices)
            center_pixel = (int(iris_center[0] * w), int(iris_center[1] * h))
            cv2.circle(annotated_image, center_pixel, 3, (0, 0, 255), -1)
        
        # Add gaze direction text
        cv2.putText(annotated_image, f"Gaze: {result['gaze_direction']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Save visualization if output path provided
        if output_path:
            cv2.imwrite(output_path, annotated_image)
        
        return result, annotated_image

# Usage example
def main():
    # Initialize the gaze tracker
    tracker = GazeTracker()
    
    # Example usage
    image_path = "WIN_20250622_21_03_41_Pro.jpg"  # Replace with your image path
    
    # Simple analysis
    result, status = tracker.analyze_gaze(image_path)
    
    if result:
        print(f"Gaze Direction: {result['gaze_direction']}")
        print(f"Left Eye - H: {result['left_eye_ratios']['horizontal']:.3f}, V: {result['left_eye_ratios']['vertical']:.3f}")
        print(f"Right Eye - H: {result['right_eye_ratios']['horizontal']:.3f}, V: {result['right_eye_ratios']['vertical']:.3f}")
        print(f"Average - H: {result['average_ratios']['horizontal']:.3f}, V: {result['average_ratios']['vertical']:.3f}")
    else:
        print(f"Error: {status}")
    
    # Analysis with visualization
    result, annotated_image = tracker.analyze_and_visualize(image_path, "output_with_landmarks.jpg")
    
    if result:
        print(f"\nVisualization saved to output_with_landmarks.jpg")
        # Optionally display the image
        # cv2.imshow('Gaze Analysis', annotated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()