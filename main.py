import cv2
import gc
import pandas as pd
import os
import google.generativeai as genai
import json

from utils import initialize_dataframe, initialize_team_df, read_video_in_batches, read_video_from_each_second, save_tracks_to_csv
from trackers import Tracker
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from player_ball_assigner import PlayerBallAssigner
from pass_detector import PassDetector
from new_data_handler import YOLOVideoProcessor
from event_process import EventProcessor
from goal_and_line_processor import GoalAndLineProcessor
from shot_detector import ShotDetector
from player_number_detector import PlayerShirtNumberTracker
from formation_detector import FormationDetector
from substitution_detector import SubstitutionDetector
from player_stats import PlayerStats
from team_stats import SoccerMatchDataProcessorFullWithSubs
from recommendation_systems import MyPlayerStats, FirstModel, SecondModel
from utils import clean_team_color, find_closest_player_dataset, find_closest_match
from utils import send_to_gemini_api_with_retry
from generate_prompt import generate_match_summary_prompt,generate_player_suggestions_prompt, generate_opponent_analysis_prompt, generate_training_suggestions_prompt

def main():
    """




                                  Start of computer vision part





    """
    # List of video file paths
    video_paths = [
        '/content/drive/MyDrive/soccergpt/videos/football4.webm'
    ]

    # Loop through each video
    for video_index, video_path in enumerate(video_paths):
        # Initialize DataFrames and persistent objects for each video
        df = initialize_dataframe()  # Initialize empty DataFrame for players
        team_df = initialize_team_df()  # Initialize empty DataFrame for teams
        tracker = Tracker('/content/drive/MyDrive/soccergpt/models/old_data.pt')  # Initialize tracker
        team_assigner = TeamAssigner()

        video_reader = cv2.VideoCapture(video_path)

        # Get total number of frames in the video
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_reader.get(cv2.CAP_PROP_FPS)
        total_seconds = int(total_frames / fps)

        # Define possible formations
        possible_formations = ['4-3-3', '4-2-3-1', '4-3-2-1', '4-1-4-1', '3-5-2', '3-4-1-2', 
                               '4-4-2', '4-4-1-1', '5-4-1', '3-4-3', '4-1-2-1-2', '3-1-4-2', 
                               '3-4-2-1', '4-5-1', '4-3-1-2', '4-2-2-2', '3-5-1-1', '4-1-3-2', 
                               '5-3-2', '3-3-3-1', '4-2-4']
        i = 0
        # Loop through the video, processing batch_size frames at a time
        for second in range(1, total_seconds + 1):
            i += 1
            # Read a batch of frames
            video_frames = read_video_from_each_second(video_reader, second)

            # If no frames were read, break the loop
            if len(video_frames) == 0:
                break

            # Process batch: Initialize per-batch objects and perform operations
            tracks = tracker.get_object_tracks(video_frames)  # Get object tracks for batch
            tracker.add_position_to_tracks(tracks)  # Add position to tracks

            # Camera movement estimator
            camera_movement_estimator = CameraMovementEstimator(video_frames[0])
            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
            camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

            # Interpolate Ball Positions
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

            # View Transformer
            view_transformer = ViewTransformer()
            view_transformer.add_transformed_position_to_tracks(tracks)

            # Speed and Distance Estimation
            speed_and_distance_estimator = SpeedAndDistance_Estimator()
            df = speed_and_distance_estimator.update_df_with_speed_and_distance(tracks, df)

            # Team Assignment
            if i == 1:
                team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
            
            for frame_num, player_track in enumerate(tracks['players']):
                for player_id, track in player_track.items():
                    team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                    tracks['players'][frame_num][player_id]['team'] = team
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
                    
                    # Update DataFrame with team and team color
                    if player_id in df.index:
                        df.at[player_id, 'team'] = team
                        df.at[player_id, 'team_color'] = str(team_assigner.team_colors[team])
                    else:
                        df.loc[player_id] = {'team': team, 'team_color': str(team_assigner.team_colors[team])}

            # Ball Assignment
            player_assigner = PlayerBallAssigner()
            for frame_num, player_track in enumerate(tracks['players']):
                if frame_num < len(tracks['ball']):
                    ball_data_for_frame = tracks['ball'][frame_num]
                    if len(ball_data_for_frame) > 0 and 1 in ball_data_for_frame:
                        ball_bbox = ball_data_for_frame[1]['bbox']
                        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                        if assigned_player != -1:
                            tracks['players'][frame_num][assigned_player]['has_ball'] = True

            # Pass Detection
            pass_detector = PassDetector(tracks, df)
            df = pass_detector.process_game_in_batches(batch_size=20)

            # YOLO Processor and Event Processing
            class_thresholds = {0: 0.8, 1: 0.7, 2: 0.3, 3: 0.1, 4: 0.7, 5: 0.6, 6: 0.85}
            yolo_processor = YOLOVideoProcessor('/content/drive/MyDrive/soccergpt/models/new_data.pt', class_thresholds)
            filtered_detections, detections_classes_2_and_3 = yolo_processor.process_frames_combined(video_frames)
            
            # Detect other events
            event_processor = EventProcessor(tracks, filtered_detections, df)
            df = event_processor.process_frames_in_batches()

            # Process Goal and Line Points
            processor = GoalAndLineProcessor()
            goals_and_lines_annotations = processor.get_goal_and_line_data(video_frames, detections_classes_2_and_3)

            # Detect shots, corners, saves, goals
            shot_detector = ShotDetector(tracks, df, team_df, goals_and_lines_annotations)
            df, team_df = shot_detector.process_frames_in_batches()

            # Initialize OCR
            player_number_tracker = PlayerShirtNumberTracker(video_frames, tracks, df,
                                           '/content/drive/MyDrive/soccergpt/models/playershirt.pt')

            # OCR: Detect player shirt numbers and update DataFrame
            df = player_number_tracker.run()

            # Initialize FormationDetector for each batch
            formation_detector = FormationDetector(tracks, possible_formations, team_df)

            # Formation Detection
            team_df = formation_detector.process_frames_in_batches()
            
            # Initialize SubstitutionDetector
            detector = SubstitutionDetector(class_thresholds, '/content/drive/MyDrive/soccergpt/models/Substitution.pt', team_df)
            # Run the extraction process
            ocr_results, team_df = detector.extract_annotation(video_frames, filtered_detections, tracks)
            
            # Delete batch-specific objects and free up memory
            del video_frames, camera_movement_estimator, view_transformer, speed_and_distance_estimator
            del player_assigner, pass_detector, yolo_processor, event_processor, processor, shot_detector
            del filtered_detections, player_number_tracker, formation_detector, detector, ocr_results

            # Force garbage collection
            gc.collect()

            # After processing all batches, fill in any missing data in DataFrames
            df = df.fillna(0)

            # Final statistics processing for teams and players
            player_stats = PlayerStats(df)
            team_1_df, team_2_df = player_stats.process_data()
        
            processor = SoccerMatchDataProcessorFullWithSubs(team_1_df, team_2_df, team_df)
            final_df = processor.process_match_data()

            # Save tracks and DataFrames to CSV files with unique names for each video
            output_suffix = f"_video_{video_index+1}"
            save_tracks_to_csv(tracks, csv_path=f'/content/Tactic_Zone/output_files_computer_vision/tracks_csv{output_suffix}.csv')
            df.to_csv(f'/content/Tactic_Zone/output_files_computer_vision/player_statistics{output_suffix}.csv', index=True)
            team_df.to_csv(f'/content/Tactic_Zone/output_files_computer_vision/team_statistics{output_suffix}.csv', index=True)
            team_1_df.to_csv(f'/content/Tactic_Zone/output_files_computer_vision/team_1_player_statistics{output_suffix}.csv', index=True)
            team_2_df.to_csv(f'/content/Tactic_Zone/output_files_computer_vision/team_2_player_statistics{output_suffix}.csv', index=True)
            final_df.to_csv(f'/content/Tactic_Zone/output_files_computer_vision/teams_final_statistics{output_suffix}.csv', index=True)

    """




    
                                  End of computer vision part

                                  Start of recommendation systems part





    """
    # Load the teams data (you may need to adjust file paths)
    teams1 = pd.read_csv('/content/Tactic_Zone/output_files_computer_vision/teams_final_statistics_video_1.csv')

    # Combine teams1 and teams2 into a single DataFrame
    combined_teams = pd.concat([teams1], ignore_index=True)

    # Load the data
    mobile_data1 = pd.read_csv('/content/Tactic_Zone/recommendation_systems_input_files/mobile_data.csv')
    mobile_data2 = pd.read_csv('/content/Tactic_Zone/recommendation_systems_input_files/mobile_data_2.csv')

    correct_shirt_numbers = [str(num) for num in mobile_data1['Shirt_Number']]
    correct_shirt_numbers2 = [str(num) for num in mobile_data2['Shirt_Number']]

    player_data1 = pd.read_csv('/content/Tactic_Zone/output_files_computer_vision/team_1_player_statistics_video_1.csv')
    player_data2 = pd.read_csv('/content/Tactic_Zone/output_files_computer_vision/team_2_player_statistics_video_1.csv')

    player_data_dict = {
        'player_data1': player_data1,
        'player_data2': player_data2
    }

    # Extract the first team's color from mobile_data1
    first_team_color_mobile1 = clean_team_color(mobile_data1.iloc[0]['Team_Color '])

    # Extract the opponent's team color from mobile_data2
    first_team_color_mobile2 = clean_team_color(mobile_data2.iloc[0]['Team_Color '])

    # Handle player data 1 and 2 based on mobile_data1
    closest_player_data_1_2 = []
    for player_data_key in ['player_data1', 'player_data2']:
        closest_player_dataset_key = find_closest_player_dataset(player_data_dict[player_data_key], first_team_color_mobile1, player_data_key)
        if closest_player_dataset_key:
            closest_player_data_1_2.append(player_data_dict[closest_player_dataset_key])

    # Combine the closest data for player 1 and 2 based on mobile_data1
    closest_player_data_mobile1 = pd.concat(closest_player_data_1_2, ignore_index=True)

    # Process the player stats
    player_stats = MyPlayerStats(closest_player_data_mobile1, correct_shirt_numbers, mobile_data1)
    closest_player_data_mobile1 = player_stats.process_data()

    # Correct the shirt numbers and drop the temporary column
    closest_player_data_mobile1['shirt_number'] = closest_player_data_mobile1['corrected_shirt_number']
    closest_player_data_mobile1.drop(columns=['corrected_shirt_number'], inplace=True)

    closest_player_data_mobile1.to_csv('/content/Tactic_Zone/output_files_recommendation_systems/closest_player_data_mobile1.csv', index=False)

    # Handle player data 3 and 4 based on mobile_data2
    closest_player_data_3_4 = []
    for player_data_key in ['player_data3', 'player_data4']:
        closest_player_dataset_key = find_closest_player_dataset(player_data_dict[player_data_key], first_team_color_mobile2, player_data_key)
        if closest_player_dataset_key:
            closest_player_data_3_4.append(player_data_dict[closest_player_dataset_key])

    # Combine the closest data for player 3 and 4 based on mobile_data2
    closest_player_data_mobile2 = pd.concat(closest_player_data_3_4, ignore_index=True)

    # Process the player stats
    player_stats = MyPlayerStats(closest_player_data_mobile2, correct_shirt_numbers2, mobile_data2)
    closest_player_data_mobile2 = player_stats.process_data()

    # Correct the shirt numbers and drop the temporary column
    closest_player_data_mobile2['shirt_number'] = closest_player_data_mobile2['corrected_shirt_number']
    closest_player_data_mobile2.drop(columns=['corrected_shirt_number'], inplace=True)

    closest_player_data_mobile2.to_csv('/content/Tactic_Zone/output_files_recommendation_systems/closest_player_data_mobile2.csv', index=False)

    # From here on, use only closest_player_data_mobile1 in the rest of the code

    # Find the closest matching rows in teams
    closest_row_team1 = find_closest_match(combined_teams, first_team_color_mobile1)
    opponent_team_color = clean_team_color(mobile_data2.iloc[0]['Team_Color '])
    closest_row_team2 = find_closest_match(combined_teams, opponent_team_color)

    combined_closest_rows = pd.DataFrame([closest_row_team1, closest_row_team2])

    my_team = pd.DataFrame([closest_row_team1])
    opponent_team = pd.DataFrame([closest_row_team2])
    my_team.to_csv('/content/Tactic_Zone/output_files_recommendation_systems/my_team.csv', index=False)
    opponent_team.to_csv('/content/Tactic_Zone/output_files_recommendation_systems/opponent_team.csv', index=False)

    # Run the models
    teams = combined_closest_rows
    data_cleaned = pd.read_csv('/content/Tactic_Zone/recommendation_systems_input_files/data_cleaned.csv')

    # Use the closest player data based on mobile data 1
    player_data = closest_player_data_mobile1

    player_data['shirt_number'] = player_data.pop('Shirt_Number')
    player_data['pass_success'] = player_data.pop('%_pass_success')
    player_data['dribbles_success'] = player_data.pop('%_dribbles_success')
    player_data['aerial_success'] = player_data.pop('%_aerial_success')
    player_data['tackles_success'] = player_data.pop('%_tackles_success')

    # Initialize the first model
    model1 = FirstModel(teams, data_cleaned)

    # Find similar rows based on the first model
    similar_rows = model1.find_similar_rows()

    # Save recommended formations to CSV
    recommended_formations = model1.find_winning_rows(similar_rows)
    recommended_formations.to_csv('/content/Tactic_Zone/output_files_recommendation_systems/recommended_formations.csv', index=False)


    # Select the first match data row
    match_data = recommended_formations
    i = 0
    solution = False
    while i < 10 and not solution:
        input_row = match_data.iloc[i].to_dict()
        input_row['tackles_success'] = input_row.pop('tackle_success')

        # Initialize the second model to recommend a team based on input row and processed player data
        team_recommender = SecondModel(input_row, player_data)
        selected_players, team_stats = team_recommender.recommend_team()

        # If a team was successfully selected, display the results
        if selected_players is not None:
            solution = True
            selected_players['status'] = 'Starting 11'  # Add a column to indicate starting players

            # Remove the selected players and recommend substitutes
            player_shirt_number_to_remove = selected_players['shirt_number'].tolist()
            player_data_updated = player_data[~player_data['shirt_number'].isin(player_shirt_number_to_remove)]

            substitute_recommender = SecondModel(input_row, player_data_updated)
            selected_substitutes, team_stats = substitute_recommender.recommend_team()

            # If substitutes were successfully selected, display the results
            if selected_substitutes is not None:
                selected_substitutes['status'] = 'Substitute'  # Add a column to indicate substitutes

                # Combine both selected players and substitutes into a single file
                combined_team = pd.concat([selected_players, selected_substitutes], ignore_index=True)
                combined_team.to_csv('/content/Tactic_Zone/output_files_recommendation_systems/combined_team.csv', index=False)
        
            else:
                selected_players.to_csv('/content/Tactic_Zone/output_files_recommendation_systems/combined_team.csv', index=False)
        i += 1
    """




                                  End of recommendation systems part
                                  Start of LLM part




    """
    # Set the environment variable in the current notebook session
    os.environ["GEMINI_API_KEY"] = ""

    opponent_info = pd.read_csv('/content/Tactic_Zone/output_files_recommendation_systems/opponent_team.csv')
    opponent_info_str = opponent_info.to_string(index=False)

    opponent_players = pd.read_csv('/content/Tactic_Zone/output_files_recommendation_systems/closest_player_data_mobile2.csv')
    opponent_players_str = opponent_players.to_string(index=False)

    my_team_info = pd.read_csv('/content/Tactic_Zone/output_files_recommendation_systems/my_team.csv')
    my_team_info_str = my_team_info.to_string(index=False)

    my_team_players = pd.read_csv('/content/Tactic_Zone/output_files_recommendation_systems/closest_player_data_mobile1.csv')
    my_team_players_str = my_team_players.to_string(index=False)

    best_formations = pd.read_csv('/content/Tactic_Zone/output_files_recommendation_systems/recommended_formations.csv')
    best_formations_str = best_formations.to_string(index = False)

    match_players_recommendations = pd.read_csv('/content/Tactic_Zone/output_files_recommendation_systems/combined_team.csv')
    match_players_recommendations_str = match_players_recommendations.to_string(index = False)

    # Configure the Gemini API key
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Generate the match summary prompt
    match_summary_prompt = generate_match_summary_prompt(my_team_info_str, opponent_info_str)
    match_summary_json = send_to_gemini_api_with_retry(match_summary_prompt)

    if match_summary_json:
        print("Match Summary Result:")
        print(json.dumps(match_summary_json, indent=4))
    else:
        print("Failed to retrieve valid JSON for match summary.")

    # Generate the suggestions prompt
    recommendation_prompt = generate_player_suggestions_prompt(best_formations_str, match_players_recommendations_str)
    recommendation_json = send_to_gemini_api_with_retry(recommendation_prompt)

    if recommendation_json:
        print("Recommendation Result:")
        print(json.dumps(recommendation_json, indent=4))
    else:
        print("Failed to retrieve valid JSON for recommendations.")

    # Generate the opponent analysis prompt
    opponent_analysis_prompt = generate_opponent_analysis_prompt(opponent_info_str, opponent_players_str)
    opponent_analysis_json = send_to_gemini_api_with_retry(opponent_analysis_prompt)

    if opponent_analysis_json:
        print("Opponent Analysis Result:")
        print(json.dumps(opponent_analysis_json, indent=4))
    else:
        print("Failed to retrieve valid JSON for opponent analysis.")

    # Generate the training suggestions prompt
    training_suggestions_prompt = generate_training_suggestions_prompt(my_team_players_str, my_team_info_str, opponent_analysis_json)
    training_suggestions_json = send_to_gemini_api_with_retry(training_suggestions_prompt)

    if training_suggestions_json:
        # Prepare the output structure
        output = {
            "team_training_session": training_suggestions_json.get("team_training_session", ""),
            "worst_5_players_individual_sessions": training_suggestions_json.get("individual_sessions", {})[:4] 
        }

        print("Training Suggestions Result:")
        print(json.dumps(output, indent=4))
    else:
        print("Failed to retrieve valid JSON for training suggestions.")

if __name__ == '__main__':
    main()