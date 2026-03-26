import os
from pathlib import Path
import ffmpeg

def make_video_from_png_sequence(
    frames_dir: str = "frames",
    pattern: str = "final_kap_%04d.png",
    output_path: str = "final_kap_movie.mp4",
    fps: int = 5,
):

    frames_dir = Path(frames_dir)
    input_pattern = str(frames_dir / pattern)

    (
        ffmpeg
        .input(input_pattern, framerate=fps)
        .output(
            output_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )



if __name__ == "__main__":
    make_video_from_png_sequence(
        frames_dir="images",
        pattern="final_kap_%04d.png",
        output_path="final_kap_movie.mp4",
        fps=5,
    )