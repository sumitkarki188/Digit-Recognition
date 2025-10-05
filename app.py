from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import os
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf

app = Flask(__name__)

# Global model variable
model = None

def train_cnn_model():
    """Train CNN model on MNIST dataset when app starts"""
    global model
    
    try:
        print("=" * 60)
        print("TRAINING CNN MODEL ON STARTUP")
        print("=" * 60)
        
        # Load MNIST dataset
        print("\n[1/4] Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', parser='auto')
        
        # Use subset for faster training (30k train, 10k test)
        print("[2/4] Preparing data...")
        x_train = mnist['data'].iloc[:30000].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train = mnist['target'].iloc[:30000].values.astype(np.int32)
        x_test = mnist['data'].iloc[60000:70000].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_test = mnist['target'].iloc[60000:70000].values.astype(np.int32)
        
        print(f"   Train shape: {x_train.shape}")
        print(f"   Test shape: {x_test.shape}")
        
        # Build CNN model
        print("\n[3/4] Building CNN architecture...")
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fully connected layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("   Model architecture created!")
        
        # Train model
        print("\n[4/4] Training model (this will take 5-10 minutes)...")
        print("   Please wait while the model trains...")
        
        history = model.fit(
            x_train, y_train, 
            validation_data=(x_test, y_test), 
            epochs=10,
            batch_size=128, 
            verbose=2  # Less verbose output
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Final Test Accuracy: {accuracy*100:.2f}%")
        print(f"Final Test Loss: {loss:.4f}")
        print("=" * 60)
        print("MODEL READY FOR PREDICTIONS!")
        print("=" * 60)
        
        return model
        
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nFailed to train model. Application may not work correctly.")
        return None

# Train model when module loads (before Gunicorn starts workers)
print("\n🚀 Starting application initialization...")
model = train_cnn_model()

if model is None:
    print("\n⚠️  WARNING: Model training failed! Predictions will not work.")
else:
    print("\n✓ Application ready to serve predictions!")

def preprocess_image(image_data):
    """Preprocess uploaded image for CNN prediction"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Apply adaptive thresholding for better digit extraction
        img = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # Find contours to crop the digit
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (the digit)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            # Crop the digit
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
        
        # Ensure non-zero dimensions
        new_h = max(new_h, 1)
        new_w = max(new_w, 1)
        
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center in 28x28 image
        final_img = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
        
        # Normalize to [0, 1]
        final_img = final_img.astype('float32') / 255.0
        
        # Reshape for CNN input: (1, 28, 28, 1)
        final_img = final_img.reshape(1, 28, 28, 1)
        
        return final_img
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        # Return blank image as fallback
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not trained yet. Please wait a few minutes and refresh.'
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
        
        # Log prediction
        print(f"Prediction: {predicted_digit}, Confidence: {confidence:.2f}%")
        
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
            'error': f'Prediction failed: {str(e)}'
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
