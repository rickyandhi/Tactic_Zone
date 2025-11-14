import csv 
import pandas as pd
import subprocess
import sys
import cv2
import firebase_admin
from firebase_admin import credentials, db

def save_tracks_to_csv(tracks, csv_path='output_tracks.csv'):
    """
    Save the tracking data to a CSV file.
    :param tracks: Dictionary containing the tracks for players, referees, ball, and goalkeepers
    :param csv_path: The file path for the output CSV
    """
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the CSV header
        writer.writerow(['Frame Number', 'Class Label', 'Track ID', 'Bounding Box (x1, y1, x2, y2)'])
        # Iterate over the tracks for each frame and write to CSV
        for frame_num, (players, referees, ball, goalkeepers) in enumerate(
            zip(tracks['players'], tracks['referees'], tracks['ball'], tracks['goalkeepers'])):
            # Write player tracks
            for track_id, track_info in players.items():
                bbox = track_info['bbox']
                writer.writerow([frame_num, 'player', track_id, bbox])
            # Write referee tracks
            for track_id, track_info in referees.items():
                bbox = track_info['bbox']
                writer.writerow([frame_num, 'referee', track_id, bbox])
            # Write ball tracks
            for track_id, track_info in ball.items():
                bbox = track_info['bbox']
                writer.writerow([frame_num, 'ball', track_id, bbox])

            # Write goalkeeper tracks
            for track_id, track_info in goalkeepers.items():
                bbox = track_info['bbox']
                writer.writerow([frame_num, 'goalkeeper', track_id, bbox])


def initialize_dataframe(tracks=None):
    """
    Initializes a DataFrame for individual player statistics.
    
    If tracking data is provided, the DataFrame is initialized with unique track IDs.
    If no tracking data is provided, an empty DataFrame with the appropriate columns is returned.
    
    :param tracks: Optional tracking data with players and their actions.
    :return: Pandas DataFrame with player stats initialized to zeros or empty with columns.
    """
    # Define the column names for player statistics
    columns = [
        "shirt_number", "Position", "Goals", "Assists", "Total_shots", 
        "Shots_on_Target", "Shots_off_Target", "Blocked_shots", 
        "Saved_shots", "Total_passes", "pass_failure", 
        "Pass_Success", "Key_passes", "dribble_attempt", 
        "dribble_failure", "dribble_success", "offensive_failure", 
        "defensive_failure", "offensive_success", 
        "defensive_success", "Tackles_attempted", 
        "tackling_failure", "tackling_success", "Clearances", 
        "Interceptions", "injuries", "Distance_covered", 
        "Avg_speed", "Highest_speed", "dribbled_past",
        "team", "team_color"
    ]

    # Check if tracks is provided
    if tracks:
        unique_track_ids = set()

        # Extract unique player IDs from tracks
        for frame_num, players in enumerate(tracks['players']):
            for track_id in players.keys():
                unique_track_ids.add(track_id)

        # Initialize DataFrame with unique player IDs as index
        df = pd.DataFrame(index=list(unique_track_ids), columns=columns)
    else:
        # Initialize an empty DataFrame with only the columns
        df = pd.DataFrame(columns=columns)

    return df


def initialize_team_df():
    """
    Initializes a DataFrame with rows for teams 1 and 2, and a single column 'corners'.
    :return: A Pandas DataFrame with team IDs and 'corners' column initialized to 0.
    """
    data = {'corners': [0, 0],'formations': [0, 0],
            'substitution_1': 0,'substitution_2': 0,'substitution_3': 0,
            'substitution_4': 0,'substitution_5': 0}  # Initialize corners to 0 for both teams
    team_df = pd.DataFrame(data, index=[1, 2])  # Create the DataFrame with team IDs as index
    return team_df

def read_video_in_batches(video_reader, start_frame, batch_size):
    """
    Reads a batch of frames from a video.

    :param video_reader: cv2.VideoCapture object used to read the video.
    :param start_frame: The starting frame number for the batch.
    :param batch_size: The number of frames to read in each batch.
    :return: A list of video frames.
    """
    # Set the starting frame position for the batch
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(batch_size):
        ret, frame = video_reader.read()
        if not ret:
            break
        frames.append(frame)
    
    return frames

def read_video_from_each_second(video_reader, second):
    frames = []

    offsets = [0, 0.25, 0.5, 0.75]

    for o in offsets:
        
        timestamp = (second + o) * 1000  # ms
        video_reader.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        ret, frame = video_reader.read()
        
        if not ret:
            continue
        
        frames.append(frame)

    return frames

def download_models():
    subprocess.run(["sed", "-i", "s/\r$//", "download_models.sh"])
    # Run the bash script
    subprocess.run(["bash", "download_models.sh"])

# Initialize Firebase Admin SDK
def initialize_firebase():
    # Replace 'path/to/your-firebase-adminsdk.json' with the path to your Firebase credentials JSON file
    cred = credentials.Certificate("fire_base/fire_base.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://tactic-zone-default-rtdb.firebaseio.com/'  # Replace with your Firebase Realtime Database URL
    })

def update_firebase_api_url(short_url):
    # Reference the Realtime Database and set the 'API_URL' value
    ref = db.reference('/')
    ref.update({'API_URL': short_url})
