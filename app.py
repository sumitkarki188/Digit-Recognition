from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import os

app = Flask(__name__)

# Global variable to store the loaded model
model = None

def create_model_architecture():
    """Create the model architecture"""
    input_layer = keras.Input(shape=(28, 28, 1), name='digit_input')
    
    x = layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1')(input_layer)
    x = layers.BatchNormalization(name='batch_norm_1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='batch_norm_2')(x)
    x = layers.MaxPooling2D((2, 2), name='maxpool_1')(x)
    x = layers.Dropout(0.25, name='dropout_1')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_3')(x)
    x = layers.BatchNormalization(name='batch_norm_3')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='batch_norm_4')(x)
    x = layers.MaxPooling2D((2, 2), name='maxpool_2')(x)
    x = layers.Dropout(0.25, name='dropout_2')(x)
    
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='batch_norm_5')(x)
    x = layers.Dropout(0.5, name='dropout_3')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='batch_norm_6')(x)
    x = layers.Dropout(0.5, name='dropout_4')(x)
    output_layer = layers.Dense(10, activation='softmax', name='digit_output')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='digit_recognition_model')
    return model

def load_model_once():
    """Load the pre-trained model once when the application starts"""
    global model
    
    try:
        print("=" * 50)
        print("INITIALIZING DIGIT RECOGNITION APP")
        print("=" * 50)
        print("Loading pre-trained model...")
        
        # Try to load SavedModel first
        savedmodel_path = 'digit_model_savedmodel'
        weights_path = 'digit_model_weights.h5'
        
        if os.path.exists(savedmodel_path):
            print("Loading SavedModel...")
            model = tf.keras.models.load_model(savedmodel_path)
            print("SavedModel loaded successfully!")
        elif os.path.exists(weights_path):
            print("SavedModel not found, loading weights instead...")
            # Create model architecture and load weights
            model = create_model_architecture()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model.load_weights(weights_path)
            print("Model weights loaded successfully!")
        else:
            print(f"ERROR: No model files found!")
            print(f"Looking for: {savedmodel_path} or {weights_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')}")
            raise FileNotFoundError("No model files found")
        
        # Test the model
        test_input = np.zeros((1, 28, 28, 1))
        _ = model.predict(test_input, verbose=0)
        print("Model test prediction successful!")
        print("=" * 50)
        print("Model ready! Application started.")
        print("=" * 50)
        
        return model
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

# Load model when the module is imported
print("Starting application initialization...")
load_model_once()
print("Application initialization complete!")

def preprocess_image(image_data):
    """Enhanced preprocessing for better accuracy"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            digit = img[y:y+h, x:x+w]
        else:
            digit = img
        
        # Resize maintaining aspect ratio
        h, w = digit.shape
        if h > w:
            new_h = 20
            new_w = int(20 * w / h) if w > 0 else 20
        else:
            new_w = 20
            new_h = int(20 * h / w) if h > 0 else 20
        
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center in 28x28
        final_img = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
        
        # Normalize
        final_img = final_img.astype('float32') / 255.0
        final_img = final_img.reshape(1, 28, 28, 1)
        
        return final_img
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            print("ERROR: Model is None when trying to predict")
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please contact administrator.'
            })
        
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image data received'
            })
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit]) * 100
        
        # Get all probabilities
        probabilities = {str(i): float(prediction[0][i] * 100) for i in range(10)}
        
        print(f"Prediction successful: digit={predicted_digit}, confidence={confidence:.2f}%")
        
        return jsonify({
            'success': True,
            'digit': int(predicted_digit),
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # This block is only for local development
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    
