from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import numpy as np

import cv2
import mediapipe as mp

class GymFormApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.camera = Camera(index=0, resolution=(640, 480), play=True)
        self.button = Button(text='Analyze Form', size_hint=(1, 0.1))
        self.button.bind(on_press=self.analyze_form)

        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.button)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS, adjust as needed

        return self.layout
    def analyze_form(self, instance):
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


if __name__ == '__main__':
    GymFormApp().run()