import cv2
import numpy as np
import pandas as pd
import os
import json

from utils import read_video_in_batches
from trackers import Tracker
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer

def calculate_distances(tracks, frame_num):
    """
    Calculates distances between players in a specific frame.
    Returns a list of dictionaries containing player distance data.
    """
    player_distances = []
    
    if 'players' not in tracks or frame_num >= len(tracks['players']):
        return player_distances

    players_in_frame = tracks['players'][frame_num]
    
    # Filter players who have a transformed position (real-world coordinates)
    # If not available, fall back to standard position (pixels) but prefer transformed
    valid_players = {}
    for p_id, p_data in players_in_frame.items():
        if 'position_transformed' in p_data:
            valid_players[p_id] = {
                'pos': np.array(p_data['position_transformed']),
                'team': p_data.get('team', -1),
                'team_color': p_data.get('team_color', None)
            }
        elif 'position' in p_data:
             # Fallback to pixel position if transformation failed, but note unit difference
             valid_players[p_id] = {
                'pos': np.array(p_data['position']),
                'team': p_data.get('team', -1),
                'team_color': p_data.get('team_color', None)
            }

    for p_id, p_info in valid_players.items():
        current_pos = p_info['pos']
        current_team = p_info['team']
        
        teammate_distances = {}
        opponent_distances = {}
        
        for other_id, other_info in valid_players.items():
            if p_id == other_id:
                continue
                
            other_pos = other_info['pos']
            other_team = other_info['team']
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(current_pos - other_pos)
            
            # Calculate angle in degrees
            # atan2(y, x) returns angle in radians between -pi and pi
            delta_y = other_pos[1] - current_pos[1]
            delta_x = other_pos[0] - current_pos[0]
            angle_rad = np.arctan2(delta_y, delta_x)
            angle_deg = np.degrees(angle_rad)
            
            # Normalize angle to be 0-360
            angle_deg = (angle_deg + 360) % 360
            
            # Round for cleaner output
            distance = round(float(distance), 2)
            angle_deg = round(float(angle_deg), 2)
            
            data_point = {'dist': distance, 'angle': angle_deg}
            
            if current_team != -1 and other_team != -1:
                if current_team == other_team:
                    teammate_distances[other_id] = data_point
                else:
                    opponent_distances[other_id] = data_point
            else:
                # If team is unknown, maybe categorize as 'unknown' or just ignore team logic
                pass

        player_distances.append({
            'frame': frame_num,
            'player_id': p_id,
            'team': current_team,
            'position_x': float(current_pos[0]),
            'position_y': float(current_pos[1]),
            'teammate_data': json.dumps(teammate_distances),
            'opponent_data': json.dumps(opponent_distances)
        })

    return player_distances

def main():
    # --- Configuration ---
    # Update this path to your video file
    video_path = '/content/drive/MyDrive/soccergpt/videos/football4.webm' 
    model_path = '/content/drive/MyDrive/soccergpt/models/old_data.pt'
    output_csv_path = '/content/drive/MyDrive/soccergpt/result/v1/player_distances.csv'
    batch_size = 200
    # ---------------------

    if not os.path.exists(video_path):
        print(f"Warning: Video path '{video_path}' does not exist. Please update the video_path variable.")
        return
        # For demonstration, we won't exit, but the cv2.VideoCapture will likely fail or open nothing.

    # Initialize modules
    tracker = Tracker(model_path)
    team_assigner = TeamAssigner()
    view_transformer = ViewTransformer()
    
    # We need to read the first frame to initialize camera movement estimator
    video_reader = cv2.VideoCapture(video_path)
    ret, first_frame = video_reader.read()
    if not ret:
        print("Failed to read video.")
        return
        
    camera_movement_estimator = CameraMovementEstimator(first_frame)
    video_reader.release() # Release to reset or just re-open in loop

    video_reader = cv2.VideoCapture(video_path)
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    target_fps = 4
    frame_interval = int(fps / target_fps)
    if frame_interval == 0:
        frame_interval = 1 # Fallback if fps is very low
    
    all_distance_data = []
    
    print(f"Processing {total_frames} frames...")
    print(f"Video FPS: {fps}. Sampling every {frame_interval} frames (approx {target_fps} FPS).")

    for start_frame in range(0, total_frames, batch_size):
        print(f"Processing batch starting at frame {start_frame}")
        video_frames = read_video_in_batches(video_reader, start_frame, batch_size)
        
        if len(video_frames) == 0:
            break

        # 1. Tracking
        tracks = tracker.get_object_tracks(video_frames)
        tracker.add_position_to_tracks(tracks)

        # 2. Camera Movement
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

        # 3. View Transformation (Pixels -> Real World)
        view_transformer.add_transformed_position_to_tracks(tracks)

        # 4. Team Assignment
        # Assign colors based on the first valid frame of the first batch
        if start_frame == 0:
            team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        
        for frame_idx, player_track in enumerate(tracks['players']):
            # frame_idx is relative to the batch
            current_frame_img = video_frames[frame_idx]
            
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(current_frame_img, track['bbox'], player_id)
                tracks['players'][frame_idx][player_id]['team'] = team
                tracks['players'][frame_idx][player_id]['team_color'] = team_assigner.team_colors[team]

        # 5. Calculate Distances
        for frame_idx in range(len(video_frames)):
            # Absolute frame number
            abs_frame_num = start_frame + frame_idx
            
            if abs_frame_num % frame_interval == 0:
                distances = calculate_distances(tracks, frame_idx)
                # Update frame number to absolute
                for d in distances:
                    d['frame'] = abs_frame_num
                
                all_distance_data.extend(distances)

    # Save results
    df = pd.DataFrame(all_distance_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Distance data saved to {output_csv_path}")

if __name__ == '__main__':
    main()
