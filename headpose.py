import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def draw_face_landmarks_fp(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

def getCoordinates_fp(face_landmarks, img_h, img_w):
    face_3d = []
    face_2d = []
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    return face_2d, face_3d, nose_2d, nose_3d

def projectCameraAngle_fp(face_2d, face_3d, img_h, img_w):
    # Fixed camera matrix (swapped img_h and img_w)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                           [0, focal_length, img_h / 2],
                           [0, 0, 1]])
    
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    return x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix

def getHeadTilt_fp(x, y, z):
    # More sensitive thresholds for left/right detection
    # Captures 45-degree angles between forward and left/right
    if y < -3:  # Changed from -20 to -10 for more sensitivity
        tiltPose = "Right"
    elif x > 14:
        tiltPose = "Up"

    elif y > 7:  # Changed from 15 to 10 for more sensitivity
        tiltPose = "Left"
    elif x < -5:
        tiltPose = "Down"
    
        
    else:
        tiltPose = "Forward"
    return tiltPose

def draw_nose_projection_fp(image, x, y, nose_2d, nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix):
    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    cv2.line(image, p1, p2, (255, 0, 0), 3)

def draw_head_tilt_pose_fp(image, text):
    cv2.putText(image, f"HEAD: {text}", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

def draw_head_tilt_angle_fp(image, x, y, z):
    cv2.putText(image, "x: " + str(np.round(x, 2)), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, "y: " + str(np.round(y, 2)), (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, "z: " + str(np.round(z, 2)), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def pipelineHeadTiltPose(image, face_landmarks):
    img_h, img_w, img_c = image.shape
    face_2d, face_3d, nose_2d, nose_3d = getCoordinates_fp(face_landmarks, img_h, img_w)
    x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix = projectCameraAngle_fp(face_2d, face_3d, img_h, img_w)
    head_pose = getHeadTilt_fp(x, y, z)
    
    # Optional: Draw visualizations
    draw_nose_projection_fp(image, x, y, nose_2d, nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    draw_head_tilt_pose_fp(image, head_pose)
    draw_head_tilt_angle_fp(image, x, y, z)
    
    return head_pose

def analyze_head_pose_from_image(image_path):
    """
    Analyze head pose from a JPG image
    Returns: "Left", "Right", "Up", "Down", "Forward", or "No face detected"
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return "Could not load image"
    
    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,  # Important for single images
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            head_pose = pipelineHeadTiltPose(image, face_landmarks)
            

            
            return head_pose
        else:
            return "No face detected"

# Usage example
if __name__ == "__main__":
    # Replace with your image path
    image_path = "WIN_20250622_21_03_41_Pro.jpg"
    result = analyze_head_pose_from_image(image_path)
    print(f"Head pose: {result}")