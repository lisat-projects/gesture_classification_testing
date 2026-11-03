import cv2
from ultralytics import YOLO
import sys
import gesture_classification_utils
import csv
import numpy as np
import tensorflow as tf

classifier = tf.keras.models.load_model("./classifier.keras")
model = YOLO("yolo26n-pose.pt")

LABELS = {
    -1: "UNLABELED",
    0: "NONE",
    1: "FOLLOW",
    2: "STAY",
    3: "FETCH",
    4: "COME",
    5: "GIVE",
    6: "INVALID",
}


def run_webcam(capture=False):
    # Open the default camera
    cam = cv2.VideoCapture(1) # if this causes issues, use cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    writer = None
    if capture:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('webcam_testing_output.mp4', fourcc, 12.0, (frame_width, frame_height))
        file = open("results.csv", "a")
        writer = csv.writer(file)

    
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # annotate frame using model
        results = model(frame, conf=0.5, verbose=False) # confidence of 0.5
        annotated_frame = results[0].plot()
        predicted_gesture = "None"
        processed_keypoints = gesture_classification_utils.process_keypoints_for_classification(results[0].keypoints)
        
        # classifier
        if processed_keypoints is not None:
            keypts_only = gesture_classification_utils.normalize_figure(results[0].keypoints).ravel()
            keypts_only = np.expand_dims(keypts_only, axis=0)
            prediction = np.argmax(classifier.predict(keypts_only), axis=1)
            predicted_gesture = LABELS[prediction[0]]
            
        annotated_frame = cv2.putText(annotated_frame, predicted_gesture, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        
        
        if capture:
            out.write(annotated_frame)
            if processed_keypoints is not None: # None indicates that body keypoints are not visible
                writer.writerow(np.append(processed_keypoints, 4)) # add class/label number to the end
        
        # Display the captured frame
        cv2.imshow('Camera', annotated_frame)

        # Press 'q' in webcam window to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    if capture:
        out.release()
        file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # run python3 webcam.py --capture
    # or python3 webcam.py to not write to file
    if ("--capture" in sys.argv):
        run_webcam(capture=True)
    else:
        run_webcam(capture=False)
        