from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("model/yolo.pt")

# image = cv2.imread("079014600_1550652054-20190220-Transjakarta-Rencanakan-Pasang-510-Kamera-E-TLE-di-Seluruh-Halte-antonius-1_jpeg.rf.e44fb5ce8703b97aa4e49e322cd3ef4c.jpg")

cap = cv2.VideoCapture("test/vecteezy_jakarta-indonesia-2021-traffic-timelapse-at-fly-over-pancoran_9161211_287.mp4")
labels = ["mobil", "motor", "truck"]
while True:
    ret, frame = cap.read()
    results = model.predict(frame)
    for res in results[0].boxes.boxes:
        label = labels[np.array(res[5], dtype=np.int32)]
        cv2.rectangle(frame, (int(res[0]), int(res[1])), (int(
            res[2]), int(res[3])), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (int(res[0]), int(
            res[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
