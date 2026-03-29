import cv2
from ultralytics import YOLO
import winsound
winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

def main():
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return


    class_names = model.names
    confidence_threshold = 0.5

    # Tracks whether a person was in the previous frame
    person_was_present = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]

        person_detected = False

        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                label = class_names[cls_id]

                if label == "person":
                    person_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Play sound only when person appears for the first time
        if person_detected and not person_was_present:
            winsound.Beep(1000, 500)  

        # Update state for next frame
        person_was_present = person_detected

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