import cv2
import numpy as np
from utils import read_video, save_video
from team_assigner import TeamAssigner
from trackers import Tracker
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read video
    frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize tracker
    tracker = Tracker('models/best.pt')

    # Get object tracks
    tracks = tracker.get_object_tracks(frames, 
                                        read_from_stub=True, 
                                        stub_path='stubs/track_stubs.pkl')

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # Assign players teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign ball acquisition
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    
    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object Tracks
    output_frames = tracker.draw_annotations(frames, tracks, team_ball_control)


    # Save video
    save_video(output_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()