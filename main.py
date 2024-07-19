import cv2
from utils import read_video, save_video
from team_assigner import TeamAssigner
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
    
    # Assign players teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw output
    ## Draw object Tracks
    output_frames = tracker.draw_annotations(frames, tracks)


    # Save video
    save_video(output_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()