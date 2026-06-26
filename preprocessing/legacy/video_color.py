import cv2
import os
from pathlib import Path

def video_color_preprocessor(video_path):
    source = cv2.VideoCapture(video_path)

    frame_width = int(source.get(3))
    frame_height = int(source.get(4))

    size = (frame_width, frame_height)
    fps = source.get(cv2.CAP_PROP_FPS)

    result = cv2.VideoWriter('gray.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size, 0)
    
    while True:
        ret, img = source.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result.write(gray)
    
    cv2.destroyAllWindows()
    source.release()
    video_name = Path(video_path.split(os.path.sep)[-1] + '_grayscale')
    parent = Path(video_path).parent.parent
    out_dir = parent / (video_path.split(os.path.sep)[-2] + '__grayscale')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_vid_path = out_dir / video_name
    os.rename('gray.avi', out_vid_path)

    return None, None, out_vid_path
