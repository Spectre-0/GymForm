from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen

import numpy as np

import cv2
import mediapipe as mp

# Define screens for the application
class SelectWorkoutScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = GridLayout(cols=2, size_hint_y=None, padding=10, spacing=10)
        layout.bind(minimum_height=layout.setter('height'))

        # Define the workout names
        workouts = [
            'Dumbbell Curls',
            'Dumbbell Shoulder Press',
            'Dumbbell Bent Over Row',
            'Dumbbell Lateral Raise',
            'Dumbbell Bench Press',
            'Dumbbell Fly'
        ]

        # Create buttons for each workout
        for workout in workouts:
            btn = Button(text=workout, size_hint_y=None, height=100)
            btn.bind(on_press=self.go_to_workout)
            layout.add_widget(btn)

        self.add_widget(layout)

    def go_to_workout(self, instance):
        # Switch to the workout detail screen with the selected workout
        # You might want to pass the workout name to the detail screen
        self.manager.current = 'workout_detail'

class WorkoutDetailScreen(Screen):
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
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            texture = self.frame_to_texture(frame)
            self.camera.texture = texture
    
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
