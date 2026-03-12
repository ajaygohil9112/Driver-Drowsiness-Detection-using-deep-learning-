import cv2 #imports opencv for computer vision
import os  #imports python's in build OS library
import numpy as np #imports numpy
import streamlit as st #imports streamlit for frontend
import pygame #imports pygame for audio file 
import csv  #imports csv for data logging file
import pandas as pd #imports pandas for confusion matrix and classification reports
from datetime import datetime #imports datetime for timestamp in logfiles

#Steamlit command for webpage to use wide layout
st.set_page_config(page_title="Drowsiness Detection", layout="wide")

# Import ML libraries
from tensorflow.keras.models import load_model #to load trained deep learning model
from tensorflow.keras.preprocessing.image import ImageDataGenerator #load imagedatagenerator
from sklearn.metrics import classification_report, confusion_matrix #imports tools to grade model's performance on the test data

#Section 1: ---------------------------------------------------------------------------------------------------------
@st.cache(allow_output_mutation=True) #loading function exactly once and cache the result and AI model
def load_resources():  #defines the function
    model_path = os.path.join("models", "model.h5") #constructs a file path to find saved model
    try:
        model = load_model(model_path) #load the model 
    except Exception as e:
        st.error(f"Error loading model: {e}") #shows the error if model not found 
        return None, None, None, None
    
    #Loads opencv's pre-trained Haar Cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    pygame.mixer.init() #Initializes the audio system and loads the file into memory
    try:
        sound = pygame.mixer.Sound('alarm.wav') #audio file loads
    except:
        sound = None
        
    return model, face_cascade, eye_cascade, sound

# Section 2. MAIN DETECTION LOOP ------------------------------------------------------------------------------------
def start_detection(model, face_cascade, eye_cascade, sound): #defines core funtion runs the webcam and analysis
    cap = cv2.VideoCapture(0) #uses opencv and opens webcam
    
    if not cap.isOpened(): #if webcam not found give error
        st.error("Could not find webcam! Check if it is plugged in.")
        return

    col1, col2 = st.columns(2) #splits the streamlit UI into two side by side columns
    with col1:
        st_frame = st.empty() #creates an empty placeholder to continuously push live video frames
    with col2:
        st.write("Drowsiness Score") 
        chart = st.line_chart() #creates an empty line chart in second column to track the drowsiness score over time
        
    #sets the font style on screen text and initializes the drowsiness score at zero
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    
    csv_filename = f"drowsiness_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" #generates a filename for the log file using current date and time
    
    try:
        with open(csv_filename, mode='w', newline='') as file: #opnes the csv file for writing 
            writer = csv.writer(file) 
            writer.writerow(["Time", "Eye Status", "Drowsiness Score"]) #writes header row on top of csv

            while cap.isOpened(): #starts a loop to continuously grab video frame
                ret, frame = cap.read() #captures frame from webcam
                if not ret:
                    st.warning("Failed to capture video frame.") #shows errors
                    break
                    
                height, width = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts the color frame to grayscale 

                faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25)) #scans the grayscale image and returns the coordinates
                cv2.rectangle(frame, (0,height-50), (200,height), (0,0,0), thickness=cv2.FILLED) #Draws a black bar at bottom of screen for text overlays 

                status = "Active" 
                
                for (x,y,w,h) in faces: #loops through every face found in frame
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1)
                    #crops the image down to just region of interest
                    roi_gray = gray[y:y+h, x:x+w] 
                    roi_color = frame[y:y+h, x:x+w]
                    
                    eyes = eye_cascade.detectMultiScale(roi_gray) #scans the cropped face region to find eyes

                    for (ex, ey, ew, eh) in eyes: #loops through every eye found on face
                        eye_img = roi_color[ey:ey+eh, ex:ex+ew] 
                        eye_img = cv2.resize(eye_img, (80,80)) #resizes the cropped eye image exactly 80x80 pixels
                        eye_img = eye_img / 255.0  #normalizes the pixel values from a 0-255 scale down to 0-1 scale
                        eye_img = eye_img.reshape(80, 80, 3) #reformats shape of image array to tensor batch format
                        eye_img = np.expand_dims(eye_img, axis=0) 
                        
                        prediction = model.predict(eye_img, verbose=0) #feeds processed eye image into AI model to get score
                        
                        #Array Logic
                        if prediction[0][0] > 0.50: #Checks if model more than 50% confident that eye is closed
                            status = "Closed"
                            score += 1  #if closed we increment danger score
                            cv2.putText(frame, "Closed", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
                        
                        elif prediction[0][1] > 0.50: #checks if the model is confident eye is open
                            status = "Open"
                            score -= 1  #if open we decrease danger score
                            cv2.putText(frame, "Open", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)

                if score < 0: score = 0  #Score never drops below zero
                
                cv2.putText(frame, 'Score:'+str(score), (100,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA) #renders current score and eye status into video frame

                if score > 15: #critical threshold check
                    cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thickness=10) #draws a red warning border around whole video feed
                    if sound:
                        try: sound.play()
                        except: pass

                chart.add_rows([score]) #pushes lates score integer to steamlit line chart
                
                timestamp = datetime.now().strftime("%H:%M:%S") 
                writer.writerow([timestamp, status, score])  #logs into csv file
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converts opencv BGR color to RGB format 
                st_frame.image(frame, channels="RGB") #fully annotated video frame to web app dashboard 
                
    finally:
        cap.release() #loop breaks (if user stops the app)
        st.success(f"Camera stopped. Data saved to your folder as: {csv_filename}")

#section 3. THE DATA TABLES ---------------------------------------------------------------------------------------------------------------------------
def evaluate_model_data(model): #for testing model's accuracy
    st.subheader("Model Performance Metrics")
    test_dir = r'D:\project\project2\Prepared_Data\Test'  #path to folder
    
    if not os.path.exists(test_dir):
        st.warning(f"⚠️ Could not find the folder: {test_dir}")
        return

    with st.spinner("Running dataset evaluation..."):
        test_datagen = ImageDataGenerator(rescale=1./255) #sets up image generator
        #automates process of pulling images from folder resizing them
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(80, 80),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        if test_generator.samples == 0:
            st.error("No images found in the 'Test' folder.")
            return

        predictions = model.predict(test_generator) #forces model to guess class for every single image
        y_pred = np.argmax(predictions, axis=1) #converts arrays into simple index
        y_true = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)  #calculates grid (true positives,falsepositive....)
            cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in class_labels], columns=[f"Predicted {c}" for c in class_labels]) #wraps raw matrix data in panda dataframe
            st.dataframe(cm_df)

        with col2:
            st.write("**Classification Report**")
            report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True) #calculates precision, recall, and f1-score for model
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df) #renders the pandas dataframe as clean table in the streamlit ui

#Section 4. THE APP UI -------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main(): #funtion for steamlit app
    st.title("Drowsiness Detection App 😴") #title 

    with st.spinner("Loading AI Models... This might take a minute..."):
        model, face_cascade, eye_cascade, sound = load_resources() #calls caching function to load the heavy assets

    if model is None:
        st.error("Failed to load resources.")
        return

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", ["🎥 Live Detection", "📊 Model Data"]) #creates navigation menu on the left side of the screen

    #simple routing logic that triggers start_detection() or evaluate_model_data() functions based on the user's radio button selection
    if app_mode == "🎥 Live Detection":  
        st.header("Live Webcam Detection")
        st.write("Click 'Start Camera'. Use Streamlit's default 'Stop' button in the top right to end the session.")
        if st.button("Start Camera"):
            start_detection(model, face_cascade, eye_cascade, sound)

    elif app_mode == "📊 Model Data":
        st.header("Dataset Evaluation")
        if st.button("Generate Matrix & Report"):
            evaluate_model_data(model)

if __name__ == '__main__':  #standard python construct that ensures function only runs if the script is executed directly
    main()