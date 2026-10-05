
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
DEST = Path("models/face_landmarker.task")


def main():
    if DEST.exists():
        print("already there:", DEST.resolve())
        return

    DEST.parent.mkdir(exist_ok=True)
    print("downloading", URL)
    urllib.request.urlretrieve(URL, DEST)
    print("done:", DEST.resolve(), f"({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
