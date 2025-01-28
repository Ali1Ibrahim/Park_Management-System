A real-time parking lot management solution using YOLO for car detection and Supabase for database updates. This project efficiently tracks the number of cars in a designated area and updates the data in the cloud for live monitoring and management.

Features

-Real-Time Detection: Leverages YOLO object detection to identify cars from a live camera feed.

-Custom Detection Area: Define a polygonal region to monitor specific areas for parking lot management.

-Cloud Integration: Automatically updates the car count to a Supabase database for seamless data synchronization.

-Dynamic Annotations: Annotates the detected cars and monitored region on the live camera feed.

-Easy Configuration: Adjustable camera input, detection area, and YOLO model for flexibility.


Tech Stack

Programming Language: Python

Libraries:

-cv2 (OpenCV) for video feed and image processing

-ultralytics (YOLO) for car detection

-pandas for structured data handling

-numpy for numerical operations

-supabase for cloud-based car count management

-Database: Supabase (PostgreSQL backend)

How It Works

Define the Detection Area:

Use a polygon to specify the area where cars should be detected.
Real-Time Detection:

The YOLO model detects objects in the video feed.

The system filters detections to count cars within the defined area.

Update the Cloud Database:

The detected car count is updated in the Supabase database after each frame.

Visual Feedback:

-Cars inside the defined area are highlighted, and the detection region is outlined for easy monitoring.
