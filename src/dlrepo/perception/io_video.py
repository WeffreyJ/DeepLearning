from typing import Union, Generator, Tuple
import cv2, time, numpy as np

def video_reader(src: Union[str,int]) -> Generator[Tuple[float, np.ndarray], None, None]:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Failed to open video source: {src}")
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        yield (time.time() - t0), frame  # BGR uint8
    cap.release()
