### Park live final 


# Import required libraries
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
from supabase import create_client, Client

# Supabase credentials
SUPABASE_URL = ""
SUPABASE_API_KEY = ""

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

def update_data(car_count):
    data = {
        "car_count": car_count
    }
    response = supabase.table("park").update(data).eq("id", 100).execute()

    print("Response object:", response)

    # Error checking
    if hasattr(response, "error") and response.error:
        print("Error updating data:", response.error)
    elif hasattr(response, "data") and response.data:
        print("Data updated successfully:", response.data)
    else:
        print("Unexpected response structure:", response)

def detect_cars_in_region_from_live_camera(model_path, area):
    # Load YOLO model
    model = YOLO(model_path)

    # Open live camera feed
    cap = cv2.VideoCapture(2)  # Use 0 for the default webcam, or change to the appropriate camera index

    if not cap.isOpened():
        raise IOError("Cannot open the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot retrieve frame. Exiting.")
            break

        # Resize the frame for consistency
        frame_resized = cv2.resize(frame, (1020, 600))

        # Perform inference
        results = model(frame)
        detections_df = pd.DataFrame(
            results[0].boxes.data.cpu().numpy(),
            columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"]
        )

        car_count = 0  # Reset car count for this frame
        for _, row in detections_df.iterrows():
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            obj_class = row["class"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box

            # Check if the object is a car and within the region
            if obj_class == 2.0:  # Class 2 corresponds to cars
                result2 = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result2 >= 0:  # Center point is inside the polygon
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        frame_resized,
                        "car",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )
                    car_count += 1  # Increment car count

        print(f"Cars in region: {car_count}")

        # Update Supabase database with the new car count for each frame
        update_data(car_count)

        # Annotate the region
        cv2.polylines(frame_resized, [np.array(area, np.int32)], True, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Car Detection", frame_resized)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Define the polygonal region (the area where cars should be detected)
area = [(78,180),(893,205),(956,468),(19,392)]

# Load YOLO model
model_path = "yolo11n.pt"  # Example model path

# Run the function
detect_cars_in_region_from_live_camera(model_path, area)
