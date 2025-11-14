import cv2
from trackers import Tracker

def draw_boxes_on_image(image_path, model_path, output_path='output_image.jpg'):
    """
    Detect objects in an image and draw bounding boxes.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        output_path: Path to save output image with boxes
    """
    # Initialize tracker
    tracker = Tracker(model_path)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Get detections for single frame
    frames = [image]
    tracks = tracker.get_object_tracks(frames)
    
    # Define colors for each object type
    colors = {
        'players': (0, 255, 0),      # Green
        'referees': (255, 255, 0),   # Cyan
        'ball': (0, 0, 255)          # Red
    }
    
    # Draw bounding boxes for each detected object type
    for object_type, object_tracks in tracks.items():
        if len(object_tracks) > 0:
            frame_tracks = object_tracks[0]  # Get first frame
            
            for track_id, track_info in frame_tracks.items():
                bbox = track_info['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw rectangle
                color = colors.get(object_type, (255, 255, 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{object_type} {track_id}"
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save output image
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")
    
    # Display the image
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Configuration
    IMAGE_PATH = '/content/input_image.jpg'  # Change this to your image path
    MODEL_PATH = '/content/drive/MyDrive/soccergpt/models/old_data.pt'  # Change this to your model path
    OUTPUT_PATH = '/content/drive/MyDrive/soccergpt/output_images/output_with_boxes.jpg'
    
    draw_boxes_on_image(IMAGE_PATH, MODEL_PATH, OUTPUT_PATH)
