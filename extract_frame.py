import os
import cv2
from typing import List, Tuple, Optional


def extract_frames_four_per_second(video_path: str, output_dir: Optional[str] = None,
                                    return_frames: bool = False, max_seconds: Optional[int] = None) -> Tuple[List, List]:
    """
    Extract four frames per second from a video file: at 0%, 25%, 75%, and 100% of each second.

    Args:
        video_path: Path to input video file.
        output_dir: If provided, extracted frames will be saved to this directory.
        return_frames: If True, function returns a list of loaded BGR frames as numpy arrays.
        max_seconds: Optional maximum number of seconds to process (for testing / early stop).

    Returns:
        Tuple of (frames, saved_paths)
            - frames: list of numpy arrays (empty list if return_frames is False)
            - saved_paths: list of file paths written (or empty list if output_dir is None)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 1.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames = [] if return_frames else []
    saved_paths = []

    percentages = [0.0, 0.25, 0.75, 1.0]  # 0%, 25%, 75%, 100%

    second = 0
    while True:
        if max_seconds and second >= max_seconds:
            break

        for pct in percentages:
            # Calculate frame index for this percentage of the second
            frame_time = second + pct
            frame_idx = int(round(frame_time * fps))

            if frame_idx >= total_frames:
                break

            # Seek to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            if output_dir:
                filename = f"frame_sec_{second:04d}_{int(pct*100):02d}pct.jpg"
                path = os.path.join(output_dir, filename)
                try:
                    cv2.imwrite(path, frame)
                    saved_paths.append(path)
                except Exception:
                    pass

            if return_frames:
                frames.append(frame.copy())

        else:
            # Only increment second if all percentages were processed
            second += 1
            continue
        break  # Break if we couldn't read a frame

    cap.release()
    return (frames if return_frames else [], saved_paths)


if __name__ == '__main__':
    # Simple CLI example
    import argparse

    parser = argparse.ArgumentParser(description='Extract 4 frames per second from video (0%, 25%, 75%, 100%)')
    parser.add_argument('video', help='Path to input video')
    parser.add_argument('--out', '-o', help='Output directory for frames', default='extracted_frames')
    parser.add_argument('--max', type=int, help='Max seconds to process (for testing)', default=None)
    args = parser.parse_args()

    frames, paths = extract_frames_four_per_second(args.video, output_dir=args.out, return_frames=False, max_seconds=args.max)
    print(f"Extracted {len(paths)} frames to {args.out}")
