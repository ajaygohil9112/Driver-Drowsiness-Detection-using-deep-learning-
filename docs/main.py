import cv2
import os
import numpy as np
import streamlit as st
import pygame

# We import tensorflow inside the function to speed up the app start
from tensorflow.keras.models import load_model

# --- 1. SMART LOADING (Prevents the Blank Screen) ---
@st.cache(allow_output_mutation=True)
def load_resources():
    # This function loads the heavy stuff only ONCE.
    
    # Load Model
    model_path = os.path.join("models", "model.h5")
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

    # Load Cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    # Load Sound
    pygame.mixer.init()
    try:
        sound = pygame.mixer.Sound('alarm.wav')
    except:
        sound = None
        
    return model, face_cascade, eye_cascade, sound

# --- 2. MAIN DETECTION LOOP ---
def start_detection(model, face_cascade, eye_cascade, sound):
    # Standard webcam setup
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not find webcam! Check if it is plugged in.")
        return
#----------------------------------------------------------------------
    col1, col2 = st.columns([1,1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.write("Drowsiness Score")
        chart = st.line_chart([0])
#-----------------------------------------------------------------------    
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video frame.")
            break
            
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Faces
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
        
        # Draw Score Background
        cv2.rectangle(frame, (0,height-50), (200,height), (0,0,0), thickness=cv2.FILLED)

        status = "Active" # Default status
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                # Preprocessing
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                eye_img = cv2.resize(eye_img, (80,80))
                eye_img = eye_img / 255.0
                eye_img = eye_img.reshape(80, 80, 3)
                eye_img = np.expand_dims(eye_img, axis=0)
                
                # Predict
                prediction = model.predict(eye_img, verbose=0) # verbose=0 hides log spam
                
                #color logic
                #box_color = (100, 100, 100)
                # Check Closed (Adjust 0.50 threshold if needed)
                if prediction[0][0] > 0.50: 
                    status = "Closed"
                    score += 1
                    cv2.putText(frame, "Closed", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
                    #box_color = (255, 0, 0) #red
                
                elif prediction[0][1] > 0.50: 
                    status = "Open"
                    score -= 1
                    cv2.putText(frame, "Open", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
                    #box_color = (0, 255, 0) #Green

                #Draw eye box
                #cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),box_color,2)

        if score < 0: score = 0
        
        cv2.putText(frame, 'Score:'+str(score), (100,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)

        # Alarm Trigger
        if score > 15:
            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thickness=10)
            if sound:
                try: sound.play()
                except: pass

        chart.add_rows([score])
        
        # DISPLAY IN STREAMLIT (Not cv2.imshow)
        # We convert BGR to RGB so colors look right in browser
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame, channels="RGB")
        
    cap.release()

# --- 3. THE APP UI ---
def main():
    st.set_page_config(page_title="Drowsiness Detection", layout="wide")
    st.title("Drowsiness Detection App 😴")
    st.write("Wait for the model to load, then click Start.")

    # Load resources with a spinner
    with st.spinner("Loading AI Models... This might take a minute..."):
        model, face_cascade, eye_cascade, sound = load_resources()

    if model is None:
        st.error("Failed to load resources. Check your terminal for errors.")
        return

    # Start Button
    if st.button("Start Camera"):
        start_detection(model, face_cascade, eye_cascade, sound)

if __name__ == '__main__':
    main()