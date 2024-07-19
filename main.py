import cv2
from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read video
    frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize tracker
    tracker = Tracker('models/best.pt')

    # Get object tracks
    tracks = tracker.get_object_tracks(frames, 
                                              read_from_stub=True, 
                                              stub_path='stubs/track_stubs.pkl')

    # Draw output
    ## Draw object Tracks
    output_frames = tracker.draw_annotations(frames, tracks)


    # Save video
    save_video(output_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()