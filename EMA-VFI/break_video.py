import cv2
import os 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="catvideo", type=str, choices=["catvideo", "bg3", "jojos"])
    args = parser.parse_args()

    if not os.path.exists(f'video/{args.video}/frames'):
        os.mkdir(f'video/{args.video}/frames')

    video_path = f'video/{args.video}/{args.video}.mp4'
    vidcap = cv2.VideoCapture(video_path)
    succ, img = vidcap.read()
    idx = 0
    while succ:
        cv2.imwrite(f'video/{args.video}/frames/frame_{idx}.jpg', img)
        idx += 1
        succ, img = vidcap.read()