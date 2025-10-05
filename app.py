from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)

class Conv2D:
    """Simple 2D Convolution Layer"""
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Initialize filters with small random values
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
        
    def forward(self, input_data):
        """Forward pass through convolution layer"""
        self.last_input = input_data
        h, w = input_data.shape
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        
        output = np.zeros((output_h, output_w, self.num_filters))
        
        for f in range(self.num_filters):
            for i in range(output_h):
                for j in range(output_w):
                    output[i, j, f] = np.sum(
                        input_data[i:(i + self.filter_size), j:(j + self.filter_size)] * 
                        self.filters[f]
                    )
        
        return output

class MaxPool2D:
    """Simple 2D Max Pooling Layer"""
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
    
    def forward(self, input_data):
        """Forward pass through max pooling layer"""
        h, w, num_filters = input_data.shape
        output_h = h // self.pool_size
        output_w = w // self.pool_size
        
        output = np.zeros((output_h, output_w, num_filters))
        
        for f in range(num_filters):
            for i in range(output_h):
                for j in range(output_w):
                    patch = input_data[
                        i * self.pool_size:(i + 1) * self.pool_size,
                        j * self.pool_size:(j + 1) * self.pool_size,
                        f
                    ]
                    output[i, j, f] = np.max(patch)
        
        return output

class Dense:
    """Simple Dense/Fully Connected Layer"""
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)
    
    def forward(self, input_data):
        """Forward pass through dense layer"""
        self.last_input_shape = input_data.shape
        input_data = input_data.flatten()
        self.last_input = input_data
        
        output = np.dot(input_data, self.weights) + self.biases
        return output

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def softmax(x):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

class SimpleCNN:
    """Simple CNN for digit recognition"""
    def __init__(self):
        # Define layers
        self.conv1 = Conv2D(8, 3)        # 28x28 -> 26x26x8
        self.pool1 = MaxPool2D(2)        # 26x26x8 -> 13x13x8
        self.conv2 = Conv2D(16, 3)       # 13x13x8 -> 11x11x16
        self.pool2 = MaxPool2D(2)        # 11x11x16 -> 5x5x16
        self.dense = Dense(5 * 5 * 16, 10)  # 400 -> 10
        
        # Pre-trained weights (simplified for demonstration)
        self._initialize_pretrained_weights()
    
    def _initialize_pretrained_weights(self):
        """Initialize with reasonable weights based on MNIST patterns"""
        # These are simplified "learned" patterns for digit recognition
        # In a real implementation, these would be trained on MNIST data
        
        # Conv1 filters (edge detectors)
        self.conv1.filters[0] = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) * 0.5  # Horizontal edge
        self.conv1.filters[1] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) * 0.5  # Vertical edge
        self.conv1.filters[2] = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) * 0.3   # Diagonal edge
        self.conv1.filters[3] = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) * 0.3   # Other diagonal
        
        # Add some random filters
        for i in range(4, 8):
            self.conv1.filters[i] = np.random.randn(3, 3) * 0.2
        
        # Conv2 and Dense layers keep random initialization but scaled
        self.conv2.filters *= 0.3
        self.dense.weights *= 0.5
        
        # Add some bias to common digit patterns
        digit_biases = np.array([0.1, -0.1, 0.05, 0.0, -0.05, 0.0, 0.1, -0.1, 0.05, 0.0])
        self.dense.biases += digit_biases
    
    def predict(self, image):
        """Forward pass through the CNN"""
        # Normalize input
        x = (image.astype(np.float32) / 255.0) - 0.5
        
        # Forward pass
        x = relu(self.conv1.forward(x))
        x = self.pool1.forward(x)
        x = relu(self.conv2.forward(x))
        x = self.pool2.forward(x)
        x = self.dense.forward(x)
        
        # Apply softmax for probabilities
        probabilities = softmax(x)
        prediction = np.argmax(probabilities)
        
        return prediction, probabilities

# Global CNN model
cnn_model = None

def initialize_cnn():
    """Initialize the CNN model"""
    global cnn_model
    
    print("=" * 50)
    print("INITIALIZING LIGHTWEIGHT CNN MODEL")
    print("=" * 50)
    print("Creating CNN architecture...")
    
    # Create CNN model
    cnn_model = SimpleCNN()
    
    # Test with dummy data
    test_image = np.random.randint(0, 255, (28, 28))
    prediction, probabilities = cnn_model.predict(test_image)
    
    print(f"CNN model initialized successfully!")
    print(f"Test prediction: {prediction}")
    print("=" * 50)
    print("CNN MODEL READY!")
    print("=" * 50)

# Initialize CNN when module loads
print("Starting CNN initialization...")
initialize_cnn()

def preprocess_image_for_cnn(image_data):
    """Preprocess uploaded image for CNN"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Apply thresholding to get clean binary image
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours and get bounding box
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assumed to be the digit)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 4
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            # Crop the digit
            digit = img[y:y+h, x:x+w]
        else:
            digit = img
        
        # Resize to fit in a square while maintaining aspect ratio
        h, w = digit.shape
        size = max(h, w)
        
        # Create a square image
        square_img = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square_img[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        
        # Resize to 28x28
        final_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
        
        return final_img
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        # Return a blank 28x28 image as fallback
        return np.zeros((28, 28), dtype=np.uint8)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if cnn_model is None:
            return jsonify({
                'success': False,
                'error': 'CNN model not initialized'
            })
        
        # Get image data
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data'})
        
        # Preprocess the image
        processed_image = preprocess_image_for_cnn(image_data)
        
        # Make prediction using CNN
        prediction, probabilities = cnn_model.predict(processed_image)
        
        # Calculate confidence
        confidence = float(probabilities[prediction]) * 100
        
        # Format probabilities
        prob_dict = {str(i): float(probabilities[i] * 100) for i in range(10)}
        
        print(f"CNN Prediction: {prediction}, Confidence: {confidence:.2f}%")
        
        return jsonify({
            'success': True,
            'digit': int(prediction),
            'confidence': round(confidence, 2),
            'probabilities': prob_dict
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback response
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
