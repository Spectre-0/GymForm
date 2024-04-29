import kivy
import json
from datetime import datetime


from kivy.config import Config

# Set the resizable option to 0 to disable window resizing
Config.set('graphics', 'resizable', '0')
# Set the desired fixed window size
Config.set('graphics', 'width', '430')
Config.set('graphics', 'height', str(int(430 * (16 / 9))))
kivy.require('2.2.1')  # replace with your current kivy version !
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import matplotlib.pyplot as plt
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
import time
from kivy.properties import NumericProperty
from kivy.properties import StringProperty 
import os


import numpy as np

import cv2
import mediapipe as mp

class ImageButton(ButtonBehavior, Image):
    pass

def format_workout_name(name):
    # Converts snake_case to Title Case
    return ' '.join(word.capitalize() for word in name.split('_'))


# Define screens for the application
class SelectWorkoutScreen(Screen):
    def go_to_workout(self, instance):
        workout_detail_screen = self.manager.get_screen('workout_detail')
        workout_detail_screen.current_workout = instance.source.split('/')[-1].split('.')[0]  # Extract workout name from image path
        workout_detail_screen.current_workout =format_workout_name(workout_detail_screen.current_workout)
        self.manager.current = 'workout_detail'




class WorkoutDetailScreen(Screen):
    current_workout = StringProperty('')  # Attribute to keep track of the current workout
    good_form_count = NumericProperty(0)  # Counter for "Good form" occurrences
    good_form_cooldown = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workout_label = self.ids.workout_label
        self.camera = self.ids.camera
        self.analyze_button = self.ids.analyze_button
        self.critiques_label = self.ids.critiques_label
        self.back_button = self.ids.back_button
        self.last_analysis_time = None
        self.last_landmarks = None

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        self.analysis_active = False
        self.analysis_event = None

    def view_history(self):
        if self.analysis_active:
            self.toggle_analysis()
        history_screen = self.manager.get_screen('history')
        history_screen.create_graph(self.current_workout)
        self.manager.current = 'history'
    def format_workout_name(name):
        # Converts snake_case to Title Case
        return ' '.join(word.capitalize() for word in name.split('_'))


    def toggle_analysis(self):
        if not self.analysis_active:
            # Start continuous analysis
            self.analysis_active = True
            self.analyze_button.text = 'Stop'
            self.analysis_event = Clock.schedule_interval(self.analyze_form, 1.0 / 30.0)  # Adjust FPS as needed
        else:
            # Stop continuous analysis
            self.analysis_active = False
            self.analyze_button.text = 'Analyze'
            if self.analysis_event:
                self.analysis_event.cancel()
                self.analysis_event = None
            if self.good_form_count > 0:
                self.save_workout_data()  # Save the workout data just before resetting
            self.good_form_count = 0

    def save_workout_data(self):
        # Create an entry for the current workout and count
        new_data = {
            "count": self.good_form_count,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # File path for the JSON data
        file_path = 'workout_data.json'
        
        # Check if the file exists and has content
        if os.path.exists(file_path) and os.stat(file_path).st_size != 0:
            # Read the existing data from the file
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
        
        # Add the new data to the existing workout key or create a new key if it doesn't exist
        if self.current_workout in data:
            data[self.current_workout].append(new_data)
        else:
            data[self.current_workout] = [new_data]
        
        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)




    def analyze_form(self, dt):
        frame = self.frame_from_camera()
        if frame is not None:
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark]

                # Check if enough time has passed or if the movement is significant
                current_time = time.time()
                if self.last_analysis_time is None or current_time - self.last_analysis_time >= 0.5:  
                    if self.last_landmarks is None or self.has_significant_movement(landmarks, self.last_landmarks):
                        # Check which workout is selected and analyze 
                        if self.current_workout == format_workout_name("dumbbell_shoulder_press"):
                            critique = self.analyze_shoulder_press(landmarks)
                            
                        elif self.current_workout == format_workout_name("dumbbell_curl"):
                            critique = self.analyze_dumbbell_curls(landmarks)
 
                        elif self.current_workout == format_workout_name("dumbbell_bent_over_rows"):
                            critique = self.analyze_bent_over_row(landmarks)
                            
                        elif self.current_workout == format_workout_name("dumbbell_lateral_raise"):
                            critique = self.analyze_lateral_raise(landmarks)

                            
                        elif self.current_workout == format_workout_name("dumbbell_bench_press"):
                            critique = self.analyze_bench_press(landmarks)
                            
                        elif self.current_workout == format_workout_name("dumbbell_fly"):
                            critique = self.analyze_dumbbell_fly(landmarks)
                            
                        else:
                            critique = "Select a workout to analyze your form."
                            print(self.current_workout)
                        

                      
                        self.critiques_label.text = critique
                        if critique == "Good form!":
                            current_time = time.time()
                            # Check if the cooldown has passed
                            if self.good_form_cooldown <= 0:
                                self.good_form_count += 1
                                # Set the cooldown period 
                                self.good_form_cooldown = 0.2
                            else:
                                # Reduce the cooldown based on time passed
                                time_passed = current_time - self.last_analysis_time
                                self.good_form_cooldown = max(0, self.good_form_cooldown - time_passed)
                            print(self.current_workout)
      
                        self.last_analysis_time = current_time
                        self.last_landmarks = landmarks

                # Update the texture with the annotated image
                annotated_image = frame.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                texture = self.frame_to_texture(annotated_image)
                self.camera.texture = texture


    def has_significant_movement(self, current_landmarks, previous_landmarks):
        # Calculate the difference between the current and previous landmarks
        # This is just an example; you may need to adjust the calculation based on your requirements
        difference = sum(abs(current_landmarks[i]['x'] - previous_landmarks[i]['x']) + abs(current_landmarks[i]['y'] - previous_landmarks[i]['y']) for i in range(len(current_landmarks)))

        # Define a threshold for significant movement
        threshold = 0.1  # Adjust the threshold as needed

        return difference > threshold

    def analyze_shoulder_press(self, landmarks):
        # Check vertical alignment of wrists over shoulders at the top of the press
        # Adding a small range for alignment tolerance
        if not (0.95 < self.is_aligned_vertically(landmarks[12], landmarks[16]) < 1.05) or \
        not (0.95 < self.is_aligned_vertically(landmarks[11], landmarks[15]) < 1.05):
            return "Align your wrists directly over your shoulders."

        # Check if arms are fully extended without locking the elbows
        # Providing a range for slight flexion to avoid locking
        right_arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        left_arm_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        if not 160 < right_arm_angle < 190:
            return "Fully extend your right arm at \n the top of the press \nwithout locking the elbow."
        if not 160 < left_arm_angle < 190:
            return "Fully extend your left arm at \nthe top of the press \nwithout locking the elbow."

        # Check torso position
        # A range is provided to allow for natural curvature of the spine
        if not (0.9 < self.is_torso_upright(landmarks[23], landmarks[11]) < 1.1) or \
        not (0.9 < self.is_torso_upright(landmarks[24], landmarks[12]) < 1.1):
            return "Keep torso upright and avoid arching your back."

        return "Good form!"

    def is_aligned_vertically(self, shoulder, wrist):
        # Vertical alignment means the x-coordinates should be very close at the top of the press
        # You might need to adjust the threshold based on how sensitive you want the check to be
        return abs(shoulder['x'] - wrist['x']) < 0.5  # Adjust this threshold as needed

    def is_torso_upright(self, hip, shoulder):
        # Upright torso means the shoulders should not dip and hips should not rise
        # A change in the y-coordinate would indicate leaning to one side or arching the back
        return abs(hip['y'] - shoulder['y']) < 0.5  # Adjust this threshold as needed
    def frame_from_camera(self):
        # Access the texture of the Camera widget
        texture = self.camera.texture
        if texture:
            # Extract the pixel data
            size = texture.size
            pixels = texture.pixels
            # Convert pixel data to an array
            frame = np.frombuffer(pixels, dtype='uint8').reshape(size[1], size[0], 4)
            # Convert from BGR to RGB (if necessary)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            return frame
        return None

    def frame_to_texture(self, frame):
        # Convert the frame to texture
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture
    
    def update(self, dt):
        # Get the current frame from the camera
        frame = self.frame_from_camera()
        if frame is not None:
            # Process the frame for pose detection
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw the pose annotations on the frame
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            # Convert the frame back to texture and display it in the camera widget
            texture = self.frame_to_texture(frame)
            self.camera.texture = texture

        if self.good_form_cooldown > 0:
            self.good_form_cooldown -= dt
            self.good_form_cooldown = max(0, self.good_form_cooldown)

    def go_back(self):
        # If analysis is active, stop it before going back
        if self.analysis_active:
            self.toggle_analysis()
        self.manager.current = 'select_workout'


    def calculate_angle(self, a, b, c):
        # shoulder, elbow, and wrist would be dictionaries with 'x', 'y', 'z' keys
        a = np.array([a['x'], a['y']])
        b = np.array([b['x'], b['y']])
        c = np.array([c['x'], c['y']])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    



    def analyze_dumbbell_curls(self, landmarks):
        # Check if elbows are close to the body
        if not self.is_close_to_body(landmarks[13], landmarks[11]) or not self.is_close_to_body(landmarks[14], landmarks[12]):
            return "Keep your elbows close to your body."
        
        # Check if wrists are aligned with elbows
        if not self.is_aligned_horizontally(landmarks[15], landmarks[13]) or not self.is_aligned_horizontally(landmarks[16], landmarks[14]):
            return "Keep your wrists aligned with your elbows."
        
        # Check if biceps are engaged
        right_arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        left_arm_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        if right_arm_angle > 30 or left_arm_angle > 30:  # Adjust the angle threshold as needed
            return "Engage your biceps more\n by curling the weight higher."
        
        return "Good form!"

    def analyze_bent_over_row(self, landmarks):
        # Check if the back is parallel to the ground
        spine_angle = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        if spine_angle < 40 or spine_angle > 120:  
            return "Keep your back\n parallel to the ground."

        # Check if elbows are tucked close to the body
        elbow_distance_to_body = abs(landmarks[13]['x'] - landmarks[11]['x'])
        if elbow_distance_to_body > 0.1:  # Example threshold, may need adjustment
            return "Tuck your elbows \ncloser to your body."

        # Check if the bar is pulled to the waist, not the chest
        bar_height = landmarks[15]['y']
        waist_height = landmarks[23]['y']
        if bar_height > waist_height:
            return "Pull the bar to your waist."

        return "Good form!"

    def analyze_lateral_raise(self, landmarks):
        # Check if elbows have a slight bend
        right_arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        left_arm_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        if right_arm_angle > 170 or left_arm_angle > 170:  # Adjust the angle threshold as needed
            return "Maintain a slight bend in your elbows."
        
        # Check if arms are raised to shoulder height
        if not self.is_aligned_vertically(landmarks[15], landmarks[11]) or not self.is_aligned_vertically(landmarks[16], landmarks[12]):
            return "Raise your arms to shoulder height."
        
        # Check if torso remains upright
        if not self.is_torso_upright(landmarks[23], landmarks[11]) or not self.is_torso_upright(landmarks[24], landmarks[12]):
            return "Keep your torso upright \nthroughout the movement."
        
        return "Good form!"

    def analyze_bench_press(self, landmarks):
        # Check if the bar is at chest level, indicating the bottom of the lift
        bar_at_chest_level = abs(landmarks[15]['y'] - landmarks[11]['y']) < 0.1  

        # When the bar is at chest level, check for wrist and elbow alignment
        if bar_at_chest_level:
            wrist_elbow_alignment = self.is_aligned_vertically(landmarks[15], landmarks[13]) and self.is_aligned_vertically(landmarks[16], landmarks[14])
            if not wrist_elbow_alignment:
                return "Align your wrists \ndirectly above your elbows."

            # Check if bar touches the chest lightly without bouncing
            # Assuming landmarks[15] is the wrist and landmarks[11] is the chest
            # You might need to adjust based on the height of the bar relative to the chest at the bottom of the lift
            if bar_at_chest_level and abs(landmarks[15]['y'] - landmarks[11]['y']) > 0.05:  # Adjust for sensitivity of touch
                return "Gently touch the bar to\n your chest without bouncing."

        # Check for proper elbow angle throughout the lift to ensure the bar path is straight
        right_elbow_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        left_elbow_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        # This checks for a straight bar path in the range of 75 to 105 degrees, which allows for some natural variation in form
        if not bar_at_chest_level and not (75 < right_elbow_angle < 105 and 75 < left_elbow_angle < 105):
            return "Maintain a straight bar \npath and a consistent elbow angle."

        return "Good form!"

    def analyze_dumbbell_fly(self, landmarks):
        # Check if elbows maintain a slight bend consistently
        right_arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        left_arm_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        # Ensuring that the arms are not locked straight and have a slight bend
        if right_arm_angle > 180 or right_arm_angle < 160 or left_arm_angle > 180 or left_arm_angle < 160:  # Adjust angles for a slight bend
            return "Maintain a consistent slight \nbend in your elbows throughout the movement."

        # Check if dumbbells are lowered to approximately chest level
        # It's common for the arms to be slightly above or below chest level during the movement
        # so a range is used instead of a strict point for comparison
        chest_height = landmarks[11]['y']
        right_dumbbell_height = landmarks[15]['y']
        left_dumbbell_height = landmarks[16]['y']
        if right_dumbbell_height - chest_height > 0.1 or left_dumbbell_height - chest_height > 0.1:  # Adjust threshold for sensitivity
            return "Lower the dumbbells to \napproximately chest level."

        # Check if arms are brought together at the top without clanging the dumbbells
        # This can be checked by ensuring the dumbbells come close together but not necessarily touching
        if abs(landmarks[15]['x'] - landmarks[16]['x']) > 0.1:  # Adjust for the distance you want to maintain between dumbbells
            return "Bring your arms closer together \nat the top of the movement \nwithout clanging the dumbbells."

        return "Good form!"

    def is_close_to_body(self, elbow, shoulder):
        return abs(elbow['x'] - shoulder['x']) < 0.1   # threshold

    def is_aligned_horizontally(self, wrist, elbow):
        return abs(wrist['x'] - elbow['x']) < 0.1  # threshold

    def is_back_straight(self, shoulder, hip):
        return abs(shoulder['y'] - hip['y']) < 0.1   # threshold

    def is_close_horizontally(self, wrist_left, wrist_right):
        return abs(wrist_left['x'] - wrist_right['x']) < 0.1   # threshold
    
    def go_to_workout(self, instance):
        workout_name = instance.source.split('/')[-1].split('.')[0]
        self.current_workout = format_workout_name(workout_name)
        self.manager.current = 'workout_detail'
    

class HistoryScreen(Screen):
    def __init__(self, **kwargs):
        super(HistoryScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.plot_area = BoxLayout(orientation='vertical', size_hint=(1, 0.9))
        layout.add_widget(self.plot_area)
        
        back_button = Button(
            text="Back",
            size_hint=(1, 0.1),
            on_press=self.go_back
        )
        layout.add_widget(back_button)
        self.add_widget(layout)

    def create_graph(self, workout_type):
        # Clear the current widgets in plot_area
        self.plot_area.clear_widgets()

        # Read data from JSON file
        try:
            with open('workout_data.json', 'r') as file:
                data = json.load(file)
                workout_data = data.get(workout_type, [])
        except Exception as e:
            workout_data = []
            print("Failed to read workout data:", e)

        counts = [entry['count'] for entry in workout_data]
        dates = [datetime.strptime(entry['datetime'], "%Y-%m-%d %H:%M:%S") for entry in workout_data]

        # Set the figure size based on DPI and desired width in inches
        dpi = 100  # This can be adjusted to your needs
        figure_width_in_inches = self.width / dpi  # Use the screen's width
        figure_height_in_inches = self.plot_area.height / dpi  # Use the plot area's height

        plt.figure(figsize=(figure_width_in_inches, figure_height_in_inches), dpi=dpi)
        plt.plot(dates, counts, 'bo-')
        plt.xlabel('Date Time')
        plt.ylabel('Good Reps count')
        plt.title(f'Workout History for {workout_type}')
        plt.gcf().autofmt_xdate()

        plt.subplots_adjust(bottom=0.2, top=0.9)  

        canvas = FigureCanvasKivyAgg(plt.gcf())
        self.plot_area.add_widget(canvas)
    def go_back(self, instance):
        self.manager.current = 'workout_detail'

class GymFormApp(App):
    def build(self):
        # Set the background color for the entire app to light grey
        Window.clearcolor = (0.9, 0.9, 0.9, 1)  # RGB values for light grey

        width =430
        # The height is determined by the 9:16 aspect ratio
        height = int(width * (16 / 9))

        # Set window size and make it non-resizable
        Window.size = (width, height)
        Config.set('graphics', 'resizable', False)

        self.sm = ScreenManager()
        self.sm.add_widget(SelectWorkoutScreen(name='select_workout'))
        self.sm.add_widget(WorkoutDetailScreen(name='workout_detail'))
        self.sm.add_widget(HistoryScreen(name='history'))

        return self.sm

if __name__ == '__main__':
    GymFormApp().run()