from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import pandas as pd
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
import numpy as np


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.prev_bboxes = {}

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections.extend(detections_batch)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, path_stub=None):
        if read_from_stub and path_stub is not None and os.path.exists(path_stub):
            with open(path_stub, 'rb') as f:
                tracks = pickle.load(f)
            return tracks 

        detections = self.detect_frames(frames)
        print(f"Number of frames: {len(frames)}")
        print(f"Number of detections: {len(detections)}")

        tracks = {
            "players": [[] for _ in range(len(frames))],
            "referees": [[] for _ in range(len(frames))],
            "ball": [[] for _ in range(len(frames))]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num].append({"track_id": track_id, "bbox": bbox})
                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num].append({"track_id": track_id, "bbox": bbox})

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num].append({"track_id": 1, "bbox": bbox})

        if path_stub is not None:
            directory = os.path.dirname(path_stub)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(path_stub, 'wb') as f:
                pickle.dump(tracks, f)

        print(f"Number of tracked frames: {len(tracks['players'])}")
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None, alpha=0.1):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Interpolate only the width between previous and current bounding box dimensions
        if track_id in self.prev_bboxes:
            prev_bbox = self.prev_bboxes[track_id]
            width = int(prev_bbox[2] * (1 - alpha) + width * alpha)
        else:
            self.prev_bboxes[track_id] = (x_center, y2, width)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None and track_id != -1:
            radius = 12
            cv2.circle(frame, (x_center, y2 + 15), radius, color, cv2.FILLED)
            
            x1_text = x_center - 10

            if track_id > 99:
                x1_text -= 15
            
            cv2.putText(
                frame,                     # The image on which the text will be drawn
                f"{track_id}",             # The text to be drawn
                (int(x1_text), int(y2 + 15 + 5)),  # The coordinates of the text's starting point (top-left corner)
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX,  # The font face to be used for the text
                0.6,                        # The font scale (size) of the text
                (0, 0, 0),                  # The color of the text (in BGR format)
                1                         # The thickness of the text
            )

        # Update previous bounding box dimensions
        self.prev_bboxes[track_id] = (x_center, y2, width)

        return frame
    

    def draw_triangle(self, frame, bbox, color):
        # Get the center of the bounding box
        x, y = get_center_of_bbox(bbox)

        # Adjust the y-coordinate to position the triangle above the ball
        y -= 15

        # Define the triangle points
        triangle_points = np.array([
            [x, y],
            [x - 7, y - 15],
            [x + 7, y - 15],
        ])

        # Draw the filled triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        # Draw the triangle border
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 1)

        return frame

    def interpolate_ball_positions_improved(self, ball_positions, window_size=5):
        flattened_ball_positions = []
        for frame_ball_positions in ball_positions:
            if frame_ball_positions:
                flattened_ball_positions.append(frame_ball_positions[0]['bbox'])
            else:
                flattened_ball_positions.append([np.nan, np.nan, np.nan, np.nan])

        df_ball_positions = pd.DataFrame(flattened_ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        for col in df_ball_positions.columns:
            df_ball_positions[col] = df_ball_positions[col].interpolate(method='linear', limit=window_size, limit_direction='both')

        interpolated_ball_positions = []
        for bbox in df_ball_positions.to_numpy().tolist():
            if np.isnan(bbox).any():
                interpolated_ball_positions.append([])
            else:
                interpolated_ball_positions.append([{"track_id": 1, "bbox": bbox}])

        return interpolated_ball_positions

    def predict_next_ball_position(self, current_pos, prev_pos):
        if prev_pos is None or np.isnan(prev_pos).any() or np.isnan(current_pos).any():
            return current_pos
        return [2 * current_pos[i] - prev_pos[i] for i in range(4)]
    
    
    def interpolate_player_positions(self, player_tracks, window_size=5):
        num_frames = len(player_tracks)
        interpolated_player_tracks = [[] for _ in range(num_frames)]
        
        # Create a dictionary to store tracks for each player ID
        player_id_tracks = {}
        
        for frame_num, frame_players in enumerate(player_tracks):
            for player in frame_players:
                player_id = player['track_id']
                if player_id not in player_id_tracks:
                    player_id_tracks[player_id] = [None] * num_frames
                player_id_tracks[player_id][frame_num] = player['bbox']
        
        for player_id, tracks in player_id_tracks.items():
            # Convert None to NaN for proper interpolation
            tracks_np = np.array([(t if t is not None else [np.nan]*4) for t in tracks])
            
            df = pd.DataFrame(tracks_np, columns=['x1', 'y1', 'x2', 'y2'])
            
            # Interpolate missing values
            for col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit=window_size, limit_direction='both')
            
            interpolated_tracks = df.to_numpy().tolist()
            
            for frame_num, bbox in enumerate(interpolated_tracks):
                if not np.isnan(bbox).any():  # Check if any value in bbox is NaN
                    interpolated_player_tracks[frame_num].append({
                        "track_id": player_id,
                        "bbox": bbox,
                        "interpolated": tracks[frame_num] is None
                    })
        
        return interpolated_player_tracks
    

    def draw_triangle_player(self, frame, bbox, color):
        center_x = int((bbox[0] + bbox[2]) / 2)
        top_y = int(bbox[1])
        
        # Define the points of the regular triangle with the base on the top
        points = np.array([
            [center_x - 10, top_y - 20],  # Top-left point of the base
            [center_x + 10, top_y - 20],  # Top-right point of the base
            [center_x, top_y]             # Bottom point (at the top of the bbox)
        ])
        
        cv2.drawContours(frame, [points], 0, color, -1)

        # Draw the black outline around the triangle
        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 0), thickness=2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Load team logos
        First_logo = cv2.imread('trackers/BM_logo.png', cv2.IMREAD_UNCHANGED) 
        other_team_logo = cv2.imread('trackers/WB_logo.png', cv2.IMREAD_UNCHANGED)
        
        # Resize logos if needed
        logo_size = (40, 40)  # Adjust size as needed
        First_logo = cv2.resize(First_logo, logo_size)
        other_team_logo = cv2.resize(other_team_logo, logo_size)

        # Constants
        overlay_width, overlay_height = 800, 150
        start_x = frame.shape[1] - overlay_width - 20
        start_y = 50
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + overlay_width, start_y + overlay_height), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Calculate possession percentages
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = team_1_frames + team_2_frames
        
        if total_frames > 0:
            team_1_possession = team_1_frames / total_frames
            team_2_possession = team_2_frames / total_frames
        else:
            team_1_possession = team_2_possession = 0

        # Draw title
        cv2.putText(frame, "Ball Possession", (start_x + 10, start_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Team colors
        team_1_color = (255, 0, 0)  # Blue for Barcelona
        team_2_color = (0, 255, 0)  # Green for the other team

        # Draw team information and progress bars
        for i, (team_logo, team_color, possession) in enumerate([
            (First_logo, team_1_color, team_1_possession),
            (other_team_logo, team_2_color, team_2_possession)
        ]):
            y_offset = 70 + i * 40
            
            # Progress bar
            bar_height = 25
            bar_width = overlay_width - 220
            bar_start_x = start_x + 150
            bar_start_y = start_y + y_offset + 5
            
            # Background of progress bar
            cv2.rectangle(frame, (bar_start_x, bar_start_y), (bar_start_x + bar_width, bar_start_y + bar_height), (200, 200, 200), -1)
            
            # Filled part of progress bar
            filled_width = int(bar_width * possession)
            cv2.rectangle(frame, (bar_start_x, bar_start_y), (bar_start_x + filled_width, bar_start_y + bar_height), team_color, -1)
            
            # Border of progress bar
            cv2.rectangle(frame, (bar_start_x, bar_start_y), (bar_start_x + bar_width, bar_start_y + bar_height), (0, 0, 0), 1)

            # Team logo (to the left of the bar)
            logo_y = start_y + y_offset
            logo_x = start_x + 75 - logo_size[0] // 2
            
            # Add logo to frame
            if team_logo.shape[2] == 4:  # If logo has an alpha channel
                alpha_s = team_logo[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    frame[logo_y:logo_y+logo_size[1], logo_x:logo_x+logo_size[0], c] = (
                        alpha_s * team_logo[:, :, c] +
                        alpha_l * frame[logo_y:logo_y+logo_size[1], logo_x:logo_x+logo_size[0], c]
                    )
            else:
                frame[logo_y:logo_y+logo_size[1], logo_x:logo_x+logo_size[0]] = team_logo
            
            # Possession percentage (to the right of the bar)
            percentage_text = f"{possession*100:.1f}%"
            text_size = cv2.getTextSize(percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, percentage_text, (bar_start_x + bar_width + 10, start_y + y_offset + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Draw border around the overlay
        cv2.rectangle(frame, (start_x, start_y), (start_x + overlay_width, start_y + overlay_height), (0, 0, 0), 2)

        return frame

    
    def draw_annotations_improved2(self, video_frames, tracks, interpolated_ball_positions, interpolated_player_positions, team_ball_control, alpha=0.1):
        output_video_frames = []
        previous_ball_position = None

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            if frame_num < len(interpolated_player_positions):
                for player in interpolated_player_positions[frame_num]:
                    color = player.get("team_color", (0, 255, 255))
                    frame = self.draw_ellipse(frame, player["bbox"], color, player["track_id"], alpha)

                    if player.get('has_ball', False):
                        frame = self.draw_triangle_player(frame, player["bbox"], (255, 255, 0))

            if frame_num < len(tracks["referees"]):
                for referee in tracks["referees"][frame_num]:   
                    frame = self.draw_ellipse(frame, referee["bbox"], (255, 0, 0), None, alpha)

            # Ball drawing logic
            ball_position = None
            if frame_num < len(tracks["ball"]) and tracks["ball"][frame_num]:
                ball_position = tracks["ball"][frame_num][0]["bbox"]
            elif frame_num < len(interpolated_ball_positions) and interpolated_ball_positions[frame_num]:
                ball_position = interpolated_ball_positions[frame_num][0]["bbox"]
            elif previous_ball_position is not None:
                ball_position = previous_ball_position

            if ball_position is not None and not np.isnan(ball_position).any():
                frame = self.draw_triangle(frame, ball_position, (0, 0, 255))  # Red color
                previous_ball_position = ball_position

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames