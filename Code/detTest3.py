from pypylon import pylon
import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import genicam

# Initialize Basler camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Get node map
nodemap = camera.GetNodeMap()

# --- Create Trackbars for Gamma and Gain ---
def nothing(x):
    pass

cv2.namedWindow("Controls")
cv2.createTrackbar("Gamma (x0.01)", "Controls", 45, 400, nothing)  # default 0.45
cv2.createTrackbar("Gain (dB*10)", "Controls", 0, 238, nothing)   # default 0.0

# Set pixel format to Mono8 if available
pixel_format = nodemap.GetNode("PixelFormat")
available_formats = []
for entry in pixel_format.GetEntries():
    symbolic = entry.GetSymbolic()
    try:
        pixel_format.SetValue(symbolic)
        available_formats.append(symbolic)
    except:
        pass

if "Mono8" in available_formats:
    pixel_format.SetValue("Mono8")
else:
    print("Mono8 not available. Exiting.")
    camera.Close()
    exit()

# Start grabbing images
camera.StartGrabbing()

# Setup image converter
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_Mono8
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Load YOLO model (change path to your model weights)
model = YOLO(r"C:\Users\Asus\Desktop\KUKA_Assembly\Object_Detection\Code\my_model.pt")  # CHANGE THE PATH

while camera.IsGrabbing():
    # --- Update Gamma and Gain from trackbars ---
    gamma_val = cv2.getTrackbarPos("Gamma (x0.01)", "Controls") / 100.0
    gain_val = cv2.getTrackbarPos("Gain (dB*10)", "Controls") / 10.0

    try:
        gamma_node = nodemap.GetNode("Gamma")
        if gamma_node and gamma_node.IsWritable():
            gamma_node.SetValue(gamma_val)
    except genicam.LogicalErrorException:
        pass

    try:
        gain_node = nodemap.GetNode("Gain")
        if gain_node and gain_node.IsWritable():
            gain_node.SetValue(gain_val)
    except genicam.LogicalErrorException:
        pass

    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab_result.GrabSucceeded():
        image = converter.Convert(grab_result)
        img = image.GetArray()

        # Convert grayscale to BGR for YOLO input
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # SCALE DOWN the image to simulate zooming out (20% of original size)
        scaled_img = cv2.resize(img_bgr, (0, 0), fx=0.2, fy=0.2)

        # Run YOLO inference
        results = model(scaled_img)

        # Draw bounding boxes and labels
        annotated_frame = scaled_img.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display annotated frame
        cv2.imshow("Basler Camera YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    grab_result.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
