from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
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

import numpy as np

import cv2
import mediapipe as mp

class ImageButton(ButtonBehavior, Image):
    pass

# Define screens for the application
# Define screens for the application
# Define screens for the application
class SelectWorkoutScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = GridLayout(cols=2, size_hint=(None, None), size=(400, 400), pos_hint={'center_x': 0.5, 'center_y': 0.5}, padding=10, spacing=10)
        layout.bind(minimum_height=layout.setter('height'))

        # Define the workout names and corresponding image paths
        workouts = [
            ('Dumbbell Curls', './images/dumbbell_curl.png'),
            ('Dumbbell Shoulder Press', './images/dumbbell_shoulder_press.jpg'),
            ('Dumbbell Bent Over Row', './images/dumbbell_bent_over_rows.png'),
            ('Dumbbell Lateral Raise', './images/dumbbell_lateral_raise.png'),
            ('Dumbbell Bench Press', './images/dumbbell_bench_press.png'),
            ('Dumbbell Fly', './images/dumbbell_fly.png')
        ]

        # Create image buttons for each workout
        for workout, image_path in workouts:
            btn = ImageButton(source=image_path, size_hint=(None, None), size=(150, 150))
            btn.bind(on_press=self.go_to_workout)
            layout.add_widget(btn)

        self.add_widget(layout)

    def go_to_workout(self, instance):
        workout_detail_screen = self.manager.get_screen('workout_detail')
        workout_detail_screen.current_workout = instance.source.split('/')[-1].split('.')[0]  # Extract workout name from image path
        self.manager.current = 'workout_detail'





class WorkoutDetailScreen(Screen):
    current_workout = None  # Attribute to keep track of the current workout
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.camera = Camera(index=0, resolution=(640, 480), play=True)
        self.analyze_button = Button(text='Analyze', size_hint=(1, 0.1))
        self.analyze_button.bind(on_press=self.toggle_analysis)
        self.critiques_label = Label(text='Critiques will appear here', size_hint=(1, 0.1))

        self.back_button = Button(text='Back', size_hint=(1, 0.1))
        self.back_button.bind(on_press=self.go_back)
        
        
        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.analyze_button)
        self.layout.add_widget(self.back_button)
        self.layout.add_widget(self.critiques_label)
        
        self.add_widget(self.layout)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        self.analysis_active = False
        self.analysis_event = None

    def toggle_analysis(self, instance):
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

    def analyze_form(self, dt):
        frame = self.frame_from_camera()
        if frame is not None:
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                # Draw the pose annotations on the frame
                annotated_image = frame.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Convert landmarks to a more usable structure
                landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark]
                
                # Check which workout is selected and analyze accordingly
                if self.current_workout == 'Dumbbell Shoulder Press':
                    critique = self.analyze_shoulder_press(landmarks)
                elif self.current_workout == 'Dumbbell Curls':
                    critique = self.analyze_dumbbell_curls(landmarks)
                elif self.current_workout == 'Dumbbell Bent Over Row':
                    critique = self.analyze_bent_over_row(landmarks)
                elif self.current_workout == 'Dumbbell Lateral Raise':
                    critique = self.analyze_lateral_raise(landmarks)
                elif self.current_workout == 'Dumbbell Bench Press':
                    critique = self.analyze_bench_press(landmarks)
                elif self.current_workout == 'Dumbbell Fly':
                    critique = self.analyze_dumbbell_fly(landmarks)
                else:
                    critique = "Select a workout to analyze your form."
                
                self.critiques_label.text = critique
                
                # Update the texture with the annotated image
                texture = self.frame_to_texture(annotated_image)
                self.camera.texture = texture

    def analyze_shoulder_press(self, landmarks):
        # Check vertical alignment of wrists over shoulders at the top of the press
        if not self.is_aligned_vertically(landmarks[12], landmarks[16]) or not self.is_aligned_vertically(landmarks[11], landmarks[15]):
            return "Align your wrists directly over your shoulders."

        # Check if arms are fully extended without locking the elbows
        right_arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        left_arm_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        if right_arm_angle < 170 or right_arm_angle > 190:
            return "Fully extend your right arm at the top of the press."
        if left_arm_angle < 170 or left_arm_angle > 190:
            return "Fully extend your left arm at the top of the press."

        # Check torso position
        if not self.is_torso_upright(landmarks[23], landmarks[11]) or not self.is_torso_upright(landmarks[24], landmarks[12]):
            return "Keep your torso upright and avoid arching your back."

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

    def go_back(self, instance):
        self.manager.current = 'select_workout'


    def calculate_angle(self, a, b, c):
        # Here you can convert the landmark data to your required format, if necessary
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
            return "Engage your biceps more by curling the weight higher."
        
        return "Good form!"

    def analyze_bent_over_row(self, landmarks):
        # Check if back is straight
        if not self.is_back_straight(landmarks[11], landmarks[23]) or not self.is_back_straight(landmarks[12], landmarks[24]):
            return "Keep your back straight throughout the movement."
        
        # Check if elbows are tucked close to the body
        if not self.is_close_to_body(landmarks[13], landmarks[11]) or not self.is_close_to_body(landmarks[14], landmarks[12]):
            return "Tuck your elbows close to your body."
        
        # Check if bar is pulled to the lower chest
        if not self.is_aligned_vertically(landmarks[15], landmarks[11]) or not self.is_aligned_vertically(landmarks[16], landmarks[12]):
            return "Pull the bar to your lower chest."
        
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
            return "Keep your torso upright throughout the movement."
        
        return "Good form!"

    def analyze_bench_press(self, landmarks):
        # Check if wrists are aligned over elbows and shoulders
        if not self.is_aligned_vertically(landmarks[15], landmarks[13]) or not self.is_aligned_vertically(landmarks[16], landmarks[14]):
            return "Keep your wrists aligned over your elbows and shoulders."
        
        # Check if bar touches the chest
        if not self.is_aligned_vertically(landmarks[15], landmarks[11]) or not self.is_aligned_vertically(landmarks[16], landmarks[12]):
            return "Lower the bar to touch your chest."
        
        # Check if elbows don't flare out too much
        right_arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        left_arm_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        if right_arm_angle < 45 or right_arm_angle > 75 or left_arm_angle < 45 or left_arm_angle > 75:
            return "Tuck your elbows closer to your body."
        
        return "Good form!"

    def analyze_dumbbell_fly(self, landmarks):
        # Check if elbows maintain a slight bend
        right_arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        left_arm_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        if abs(right_arm_angle - left_arm_angle) > 10:  # Adjust the angle difference threshold as needed
            return "Maintain a consistent slight bend in your elbows."
        
        # Check if dumbbells are lowered to chest level
        if not self.is_aligned_vertically(landmarks[15], landmarks[11]) or not self.is_aligned_vertically(landmarks[16], landmarks[12]):
            return "Lower the dumbbells to chest level."
        
        # Check if arms are brought together at the top
        if not self.is_close_horizontally(landmarks[15], landmarks[16]):
            return "Bring your arms together at the top of the movement."
        
        return "Good form!"

    def is_close_to_body(self, elbow, shoulder):
        return abs(elbow['x'] - shoulder['x']) < 0.1  # Adjust the threshold as needed

    def is_aligned_horizontally(self, wrist, elbow):
        return abs(wrist['x'] - elbow['x']) < 0.1  # Adjust the threshold as needed

    def is_back_straight(self, shoulder, hip):
        return abs(shoulder['y'] - hip['y']) < 0.1  # Adjust the threshold as needed

    def is_close_horizontally(self, wrist_left, wrist_right):
        return abs(wrist_left['x'] - wrist_right['x']) < 0.1  # Adjust the threshold as needed

class GymFormApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.select_workout_screen = SelectWorkoutScreen(name='select_workout')
        self.workout_detail_screen = WorkoutDetailScreen(name='workout_detail')

        self.sm.add_widget(self.select_workout_screen)
        self.sm.add_widget(self.workout_detail_screen)

        return self.sm
    


if __name__ == '__main__':
    GymFormApp().run()