import cv2
from ultralytics import YOLO

# The "Main" function. Logic for basic image detection is within this function.
def main():
    # This is the pretrained YOLO model I was talking about from the Ultralytics library.
    model = YOLO("yolo11n.pt")

    # This opens the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    class_names = model.names
    confidence_threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = model(frame, verbose=False)
        result = results[0]

        # Reset for each frame
        cellphone_detected = False

        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "cell phone":
                    cellphone_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

        # Use the boolean after checking all detections
        if cellphone_detected:
            message = "Cellphone detected!"
            color = (0, 255, 0)
        else:
            message = "No cell phone detected"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            message,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("MQAIS Object Detection Starter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()