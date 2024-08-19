import cv2
import os

def extract_frames(video_path, output_dir):
  """Extracts frames from a video and saves them to an output directory.

  Args:
    video_path: Path to the input video file.
    output_dir: Path to the output directory for frames.
  """

  cap = cv2.VideoCapture(video_path)
  count = 0

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    cv2.imwrite(os.path.join(output_dir, f"frame_{count:06d}.png"), frame)
    count += 1

  cap.release()
  cv2.destroyAllWindows()

# Example usage:
video_path = "Match_1951_1_0_subclip.mp4"
output_dir = "./extracted_frames"
extract_frames(video_path, output_dir)