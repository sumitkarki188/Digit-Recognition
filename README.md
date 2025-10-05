# ✍️ Handwritten Digit Recognition

A deep learning web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can draw digits on a canvas or upload images for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Features

- **Interactive Canvas Drawing**: Draw digits directly on the web interface
- **Image Upload**: Upload handwritten digit images for recognition
- **Real-time Predictions**: Instant digit recognition with confidence scores
- **Probability Visualization**: View prediction probabilities for all digits (0-9)
- **High Accuracy**: Achieves 98.76% accuracy on MNIST test dataset
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Demo

**Live Demo**: [https://digit-recognition-o4mx.onrender.com/](https://digit-recognition-o4mx.onrender.com/)

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow/Keras (CNN model)
- **Frontend**: HTML5, CSS3, JavaScript (Canvas API)
- **Deployment**: Render
- **Dataset**: MNIST (70,000 handwritten digit images)

## 📊 Model Architecture

Model: Sequential CNN

Layer (type) Output Shape Param #  
conv2d (Conv2D) (None, 26, 26, 64) 640  
activation (Activation) (None, 26, 26, 64) 0  
max_pooling2d (MaxPooling2D) (None, 13, 13, 64) 0  
conv2d_1 (Conv2D) (None, 11, 11, 64) 36928  
activation_1 (Activation) (None, 11, 11, 64) 0  
max_pooling2d_1 (MaxPooling) (None, 5, 5, 64) 0  
conv2d_2 (Conv2D) (None, 3, 3, 64) 36928  
activation_2 (Activation) (None, 3, 3, 64) 0  
max_pooling2d_2 (MaxPooling) (None, 1, 1, 64) 0  
flatten (Flatten) (None, 64) 0  
dense (Dense) (None, 64) 4160  
activation_3 (Activation) (None, 64) 0  
dense_1 (Dense) (None, 32) 2080  
activation_4 (Activation) (None, 32) 0  
dense_2 (Dense) (None, 10) 330  
activation_5 (Activation) (None, 10) 0  
Total params: 81,066  
Trainable params: 81,066  
Non-trainable params: 0  

**Performance Metrics:**
- Test Accuracy: **98.76%**
- Training Time: ~245 seconds (10 epochs)
- Model Size: ~1 MB

## 📁 Project Structure

digit-recognition/
├── app.py # Flask application  
├── mnist_cnn_model.h5 # Trained CNN model  
├── requirements.txt # Python dependencies  
├── templates/  
│ └── index.html # Frontend HTM  
├── static/  
│ ├── style.css # Styling  
│ └── script.js # JavaScript logic  
├── README.md # Project documentation  
└── .gitignore # Git ignore rules  

## 🔧 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Local Development Setup

1. **Clone the repository**
git clone https://github.com/YOUR_USERNAME/digit-recognition.git
cd digit-recognition

2. **Create virtual environment**
Windows
py -3.10 -m venv venv
venv\Scripts\activate

macOS/Linux
python3.10 -m venv venv
source venv/bin/activate

3. **Install dependencies**
pip install --upgrade pip
pip install -r requirements.txt

4. **Run the application**
python app.py

5. **Open in browser**
http://localhost:5000

## 📦 Dependencies

flask==3.0.0
tensorflow==2.15.0
numpy==1.24.3
pillow==10.1.0
gunicorn==21.2.0

## 🎓 Training the Model

To train the CNN model from scratch:

python train_model.py

The training script:
1. Loads the MNIST dataset (60,000 training + 10,000 test images)
2. Preprocesses images (normalization, reshaping)
3. Builds and compiles the CNN architecture
4. Trains for 10 epochs with validation
5. Saves the model as `mnist_cnn_model.h5`

**Training Results:**
Epoch 10/10
469/469 [==============================] - 24s 51ms/step
loss: 0.0156 - accuracy: 0.9951 - val_loss: 0.0392 - val_accuracy: 0.9876

## 🌐 Deployment

### Deploy to Render

1. **Push code to GitHub**
git add .
git commit -m "Ready for deployment"
git push origin main

2. **Create Render Web Service**
   - Go to [render.com](https://render.com)
   - New → Web Service
   - Connect GitHub repository
   - Configure:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Instance Type**: Free

3. **Access deployed app**
https://your-app-name.onrender.com

## 🧪 Usage Examples

### Drawing on Canvas
1. Open the application
2. Click "Draw Digit" tab
3. Draw a digit (0-9) on the canvas
4. Click "Predict"
5. View the prediction result

### Uploading Image
1. Click "Upload Image" tab
2. Select an image file (PNG, JPG)
3. Click "Predict"
4. View the prediction result

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 98.76% |
| Precision | 98.80% |
| Recall | 98.75% |
| F1-Score | 98.77% |
| Training Time | 245 seconds |

## 🐛 Known Issues

- Free tier on Render sleeps after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds to wake up
- Large model file (~1MB) increases initial load time

## 🚧 Future Enhancements

- [ ] Add support for multi-digit recognition
- [ ] Implement drawing undo/redo functionality
- [ ] Add model confidence threshold settings
- [ ] Support for digit sequences (e.g., phone numbers)
- [ ] Export predictions as JSON
- [ ] Add dark mode theme
- [ ] Optimize model for faster inference
- [ ] Add batch prediction for multiple images

