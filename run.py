
import cv2
import numpy as np
from ultralytics import YOLO

# Create a mapping of allergens, food labels, and their indices
class_indices = {
    'alcohol': (0, 'Histamine'),
    'alcohol_glass': (1, 'Histamine'),
    'almond': (2, 'Salicylate'),
    'avocado': (3, 'Histamine'),
    'blackberry': (4, 'Salicylate'),
    'blueberry': (5, 'Salicylate'),
    'bread': (6, 'Gluten'),
    'bread_loaf': (7, 'Gluten'),
    'capsicum': (8, 'Salicylate'),
    'cheese': (9, 'Lactose'),
    'chocolate': (10, 'Lactose/Caffeine'),
    'cooked_meat': (11, 'Histamine'),
    'dates': (12, 'Salicylate'),
    'egg': (13, 'Ovomucoid'),
    'eggplant': (14, 'Histamine'),
    'icecream': (15, 'Lactose'),
    'milk': (16, 'Lactose'),
    'milk_based_beverage': (17, 'Lactose/Caffeine'),
    'mushroom': (18, 'Salicylate'),
    'non_milk_based_beverage': (19, 'Caffeine'),
    'pasta': (20, 'Gluten'),
    'pineapple': (21, 'Salicylate'),
    'pistachio': (22, 'Salicylate'),
    'pizza': (23, 'Gluten'),
    'raw_meat': (24, 'Histamine'),
    'roti': (25, 'Gluten'),
    'spinach': (26, 'Histamine'),
    'strawberry': (27, 'Salicylate'),
    'tomato': (28, 'Salicylate'),
    'whole_egg_boiled': (29, 'Ovomucoid')
}

# Create a reverse mapping from index to class name and allergen
index_to_class_allergen = {index: (class_name, allergen) for class_name, (index, allergen) in class_indices.items()}

# Function to resize and pad the image while maintaining aspect ratio
def resize_and_pad(image, target_size=(416, 416)):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a new square image and place the resized image in the center
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded_image[(target_h - new_h) // 2:(target_h - new_h) // 2 + new_h,
                 (target_w - new_w) // 2:(target_w - new_w) // 2 + new_w] = resized_image

    return padded_image, scale, (new_w, new_h)  # Return scale and new dimensions for later use

# Load the trained model
model = YOLO("best.pt")  # Update with your actual path

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and pad the frame
    padded_image, scale, (new_w, new_h) = resize_and_pad(frame)

    # Run inference on the padded image
    results = model.predict(source=padded_image)

    # Draw bounding boxes on the original frame for display
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # bounding box coordinates
            conf = box.conf[0].item()  # confidence
            cls = box.cls[0].item()  # class label
            
            # Adjust the bounding box coordinates for the original frame
            x1 = int((x1 - (416 - new_w) // 2) / scale)
            y1 = int((y1 - (416 - new_h) // 2) / scale)
            x2 = int((x2 - (416 - new_w) // 2) / scale)
            y2 = int((y2 - (416 - new_h) // 2) / scale)

            # Get the class name and allergen from the reverse mapping
            class_info = index_to_class_allergen.get(int(cls), ("Unknown", "Unknown"))  # Default to ("Unknown", "Unknown")
            class_name, allergen = class_info
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, f"{class_name} ({allergen}) Conf: {conf:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

