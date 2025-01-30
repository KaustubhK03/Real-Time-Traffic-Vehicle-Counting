from Crossing_counter import RealTimeTrafficCounter
from coco_classes import class_names

model_path = "yolo11n.pt"
video_path = "../AITraffic/Traffic_Video/Front+View+Highway.mp4"
out_put_video_path = "Real_Time_output/front+view+highway.avi"
output_csv = "../AITraffic/CSVs/front+view+highway.csv"
line_coords = [(0, 1800), (4000, 1800)]  # Adjust line coordinates as needed
counter = RealTimeTrafficCounter(model_path, video_path, output_csv, out_put_video_path, line_coords)
counter.process_video()
