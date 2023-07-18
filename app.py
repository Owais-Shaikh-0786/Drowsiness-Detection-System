import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
import pygame.mixer as mixer


def run_drowsiness_detection_app():
    # Load Haar cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Load the trained model
    # model = load_model(r'C:\Users\shaik\OneDrive\Desktop\Drowsiness-Detection-System\models\model.h5')

    model_path = 'models/model.h5'  # Path relative to the root directory of your app
    model = load_model(model_path)

    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Initialize variables
    Score = 0
    closed_eyes_count = 0
    closed_eyes_threshold = 5
    is_alarm_playing = False

    st.title("Drowsiness Detection System")

    # Streamlit app loop
    session_state = st.session_state
    if 'is_running' not in session_state:
        session_state.is_running = False

    start_button = st.button("Start")
    stop_button = st.button("Stop")

    if start_button:
        session_state.is_running = True

    if stop_button:
        session_state.is_running = False
        cap.release()
        cv2.destroyAllWindows()

    while session_state.is_running:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reset the score for each frame
        Score = 0

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

        for (x, y, w, h) in faces:
            # Perform eye detection within the face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=1)

            for (ex, ey, ew, eh) in eyes:
                # Preprocess the eye region
                eye = roi_color[ey:ey+eh, ex:ex+ew]
                eye = cv2.resize(eye, (80, 80))
                eye = eye / 255.0
                eye = np.expand_dims(eye, axis=0)

                # Perform prediction using the loaded model
                prediction = model.predict(eye)

                # If eyes are closed
                if prediction[0][0] > 0.30:
                    cv2.putText(frame, 'closed', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                                color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    closed_eyes_count += 1
                    Score += 1

                    if closed_eyes_count > closed_eyes_threshold and not is_alarm_playing:
                        try:
                            mixer.init()  # Initialize Pygame mixer
                            sound_file = 'alarm.wav'
                            sound = mixer.Sound(sound_file)
                            sound.play()
                            is_alarm_playing = True
                        except:
                            pass

                # If eyes are open
                elif prediction[0][1] > 0.90:
                    cv2.putText(frame, 'open', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                                color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    closed_eyes_count = 0

        # Update the alarm state outside the loop
        if Score <= closed_eyes_threshold and is_alarm_playing:
            try:
                sound.stop()
                mixer.quit()  # Quit Pygame mixer
                is_alarm_playing = False
            except:
                pass

        cv2.putText(frame, 'Score ' + str(Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Resize the frame to full screen
        cv2.namedWindow('Drowsiness Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Drowsiness Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_drowsiness_detection_app()
