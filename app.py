from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

app = Flask(__name__)

# Global variable to store the trained model
model = None

def train_model_once():
    """Train a CNN model once when the application starts"""
    global model
    
    print("Loading MNIST dataset...")
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess the data - reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Create CNN model for better accuracy
    print("Building CNN architecture...")
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training CNN model... This will take a few minutes.")
    # Train the model with data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    datagen.fit(x_train)
    
    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Model trained! Test accuracy: {test_accuracy*100:.2f}%")
    
    return model

def preprocess_image(image_data):
    """Enhanced preprocessing for better accuracy"""
    # Decode base64 image
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply adaptive thresholding for better binarization
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours to get the digit bounding box
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Crop the digit
        digit = img[y:y+h, x:x+w]
    else:
        digit = img
    
    # Resize to fit in 20x20 box while maintaining aspect ratio
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(20 * w / h)
    else:
        new_w = 20
        new_h = int(20 * h / w)
    
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create 28x28 image with the digit centered
    final_img = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
    
    # Apply morphological operations to clean the image
    kernel = np.ones((2, 2), np.uint8)
    final_img = cv2.morphologyEx(final_img, cv2.MORPH_CLOSE, kernel)
    
    # Normalize
    final_img = final_img.astype('float32') / 255.0
    
    # Reshape for CNN (add batch and channel dimensions)
    final_img = final_img.reshape(1, 28, 28, 1)
    
    return final_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit]) * 100
        
        # Get all probabilities
        probabilities = {str(i): float(prediction[0][i] * 100) for i in range(10)}
        
        return jsonify({
            'success': True,
            'digit': int(predicted_digit),
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("=" * 50)
    print("INITIALIZING DIGIT RECOGNITION APP")
    print("=" * 50)
    train_model_once()
    print("=" * 50)
    print("Model ready! Starting Flask server...")
    print("=" * 50)
    
    # Get port from environment variable (Render uses this)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
