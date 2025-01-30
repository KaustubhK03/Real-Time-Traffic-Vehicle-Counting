# from ultralytics import YOLO
# import cv2
# import pandas as pd
# import torch
#
# # Check for MPS support and set the device
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
#
# # Load the YOLO model
# model = YOLO("../Computer_Vision/yolo/yolo11n.pt")
#
# cap = cv2.VideoCapture("../AITraffic/Traffic_Video/Traffic_Stop.mp4")
#
# all_detections = []  # List to store detections for all frames
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Break if the video ends or there's an issue
#
#     # Move the frame to the selected device (not required for YOLO models as they handle it internally)
#     # Run object detection
#     results = model(frame)  # YOLO automatically uses the correct device
#
#     # Extract detected objects
#     detections = results[0].boxes.data  # Bounding boxes and detection info
#     print(detections)
#     # Process detections
#     for idx, detection in enumerate(detections):
#         x1, y1, x2, y2, confidence, class_id = detection[:6]
#         all_detections.append({
#             "box_id": idx,  # Unique ID for each box in the current frame
#             "frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
#             "x1": float(x1),
#             "y1": float(y1),
#             "x2": float(x2),
#             "y2": float(y2),
#             "confidence": float(confidence),
#             "class_id": int(class_id)
#         })
#
#     # Optionally, draw detections on the frame
#     annotated_frame = results[0].plot()  # Annotated frame with detections
#
#     # Display the frame
#     cv2.imshow("YOLO Detection", annotated_frame)
#
#     # Break the loop on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#
# # Convert the list of detections into a pandas DataFrame
# detections_df = pd.DataFrame(all_detections)
# detections_df.to_csv('Tracking.csv')

from Crossing_counter import RealTimeTrafficCounter
from coco_classes import class_names

model_path = "yolo11n.pt"
video_path = "../AITraffic/Traffic_Video/Front+View+Highway.mp4"
out_put_video_path = "Real_Time_output/front+view+highway.avi"
output_csv = "../AITraffic/CSVs/front+view+highway.csv"
line_coords = [(0, 1800), (4000, 1800)]  # Adjust line coordinates as needed
counter = RealTimeTrafficCounter(model_path, video_path, output_csv, out_put_video_path, line_coords)
counter.process_video()
