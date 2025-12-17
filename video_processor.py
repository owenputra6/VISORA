import time
import threading
import cv2
import numpy as np
import av
from streamlit_webrtc import VideoProcessorBase
from realtime_tts import speak_q, tts_busy
import easyocr

class VideoProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, class_names, colors):
        self.model = yolo_model
        self.names = class_names
        self.colors = colors

        self._lock = threading.Lock()
        self.latest_frame = None
        self.last_drawn = None
        self.ocr_reader = easyocr.Reader(['id', 'en'], gpu=False)

        self.stop = False

        threading.Thread(
            target=self._detector_loop,
            daemon=True
        ).start()

    def _detector_loop(self):
        while not self.stop:
            with self._lock:
                if self.latest_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self.latest_frame.copy()

            results = self.model.predict(
                frame,
                imgsz=640,
                conf=0.5,
                verbose=False
            )

            labels = []
            img = frame.copy()

            if results and results[0].boxes is not None:
                for box, cls, conf in zip(
                    results[0].boxes.xyxy.cpu().numpy(),
                    results[0].boxes.cls.cpu().numpy(),
                    results[0].boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = box.astype(int)
                    label = self.names[int(cls)]
                    labels.append(label)

                    color = self.colors[int(cls) % len(self.colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            ocr_texts = []
            if labels:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                results_ocr = self.ocr_reader.readtext(
                    gray,
                    detail=0,
                    paragraph=True
                )

                ocr_texts = [t.strip() for t in results_ocr if len(t.strip()) > 2]

            y = 30
            for text in ocr_texts[:3]:
                cv2.putText(
                    img,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                y += 30

            parts = []

            if labels:
                parts.append("objek " + ", ".join(sorted(set(labels))))

            if ocr_texts:
                parts.append("teks " + ", ".join(ocr_texts))

            if labels or ocr_texts:
                sentence = "Saya melihat " + ", ".join(parts)
                speak_q.put_nowait(sentence)

            self.last_drawn = img
            time.sleep(0.2)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        with self._lock:
            self.latest_frame = img
            out = self.last_drawn if self.last_drawn is not None else img

        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def __del__(self):
        self.stop = True
