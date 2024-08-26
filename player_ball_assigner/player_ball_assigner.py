import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance
class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        self.last_assigned_player = -1
        self.frames_since_last_assignment = 0
        self.max_frames_to_keep_possession = 10  # Adjust this value as needed

    def assign_ball_to_player(self, players, ball_bbox):
        if ball_bbox is None:
            self.frames_since_last_assignment += 1
            if self.frames_since_last_assignment > self.max_frames_to_keep_possession:
                self.last_assigned_player = -1
            return self.last_assigned_player

        ball_position = get_center_of_bbox(ball_bbox)
        min_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        if assigned_player != -1:
            self.last_assigned_player = assigned_player
            self.frames_since_last_assignment = 0
        else:
            self.frames_since_last_assignment += 1
            if self.frames_since_last_assignment > self.max_frames_to_keep_possession:
                self.last_assigned_player = -1

        return self.last_assigned_player