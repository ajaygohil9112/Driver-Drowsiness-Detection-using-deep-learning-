import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load your trained model
model_path = os.path.join("models", "model.h5")
print("Loading model...")
model = load_model(model_path)

# 2. Set up the Test Data Generator
# IMPORTANT: Ensure your test dataset is organized as:
# test_data/
# ├── Closed/ (contains images of closed eyes)
# └── Open/   (contains images of open eyes)
test_dir = r'D:\project\project2\Prepared_Data\Test' 

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(80, 80), # Matches your Streamlit app's cv2.resize
    batch_size=32,
    class_mode='categorical',
    shuffle=False # Keep false so predictions match file order
)

# 3. Generate Predictions
print("Generating predictions...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# 4. Generate Raw Data Tables
print("\n--- CONFUSION MATRIX ---")
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in class_labels], columns=[f"Predicted_{c}" for c in class_labels])
print(cm_df)

print("\n--- CLASSIFICATION REPORT ---")
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Optional: Save to CSV
report_df.to_csv("model_evaluation_metrics.csv")
print("\nMetrics saved to model_evaluation_metrics.csv")