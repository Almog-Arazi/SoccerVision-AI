from utils import read_video, save_video
from trackers import Tracker
import os
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np

def main():
    try:
        # Read Video
        input_video_path = 'input_videos/game.mp4'
        print(f"Reading video from: {input_video_path}")
        video_frames = read_video(input_video_path)
        print(f"Number of video frames: {len(video_frames)}")

        # Initialize the Tracker
        model_path = 'models/best_new.pt'
        print(f"Initializing Tracker with model: {model_path}")
        tracker = Tracker(model_path)

        # Get object tracks
        stub_path = 'stubs/track_stubs.pkl'
        print(f"Getting object tracks (using stub: {os.path.exists(stub_path)})")
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, path_stub=stub_path)

        # Save cropped image of a player for debugging 
        if tracks['players'][0]:  # Check if there are any players in the first frame
            player = tracks['players'][0][0]  # Get the first player
            bbox = player['bbox']
            frame = video_frames[0]
            cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)

        # Interpolate the ball track   
        interpolated_ball_positions = tracker.interpolate_ball_positions_improved(tracks["ball"], window_size=5)
        interpolated_player_positions = tracker.interpolate_player_positions(tracks["players"], window_size=5)
       
        # Assign Player Teams
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        
        # Assign team to interpolated player positions
        for frame_num, player_track in enumerate(interpolated_player_positions): 
            for player in player_track:
                player_id = player['track_id']
                team = team_assigner.get_player_team(video_frames[frame_num], player['bbox'], player_id)
                player['team'] = team 
                player['team_color'] = team_assigner.team_colors[team]

        # Initialize PlayerBallAssigner
        player_assigner = PlayerBallAssigner()
        team_ball_control = []

        for frame_num in range(len(video_frames)):
            try:
                if frame_num < len(tracks['ball']) and len(tracks['ball'][frame_num]) > 0:
                    ball_bbox = tracks['ball'][frame_num][0]['bbox']
                else:
                    ball_bbox = None
                
                # Convert player_track list to dictionary
                player_dict = {i: player for i, player in enumerate(interpolated_player_positions[frame_num])}
                
                assigned_player = player_assigner.assign_ball_to_player(player_dict, ball_bbox)

                if assigned_player != -1 and assigned_player < len(interpolated_player_positions[frame_num]):
                    interpolated_player_positions[frame_num][assigned_player]['has_ball'] = True
                    print(f"Frame {frame_num}: Player {assigned_player} has the ball.")
                    if 'team' in interpolated_player_positions[frame_num][assigned_player]:
                        team_ball_control.append(interpolated_player_positions[frame_num][assigned_player]['team'])
                    else:
                        team_ball_control.append(-1)  # or any default value indicating no team
                else:
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
            except IndexError as e:
                print(f"An error occurred at frame {frame_num}: {str(e)}")
                import traceback
                traceback.print_exc()
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)

        team_ball_control = np.array(team_ball_control)

        # Draw output 
        print("Drawing annotations on frames")
        output_video_frames = tracker.draw_annotations_improved2(video_frames, tracks, interpolated_ball_positions, interpolated_player_positions, team_ball_control)

        # Save video
        output_path = 'output_videos/resGame.avi'
        print(f"Saving output video to: {output_path}")
        save_video(output_video_frames, output_path)

        print("Processing completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()