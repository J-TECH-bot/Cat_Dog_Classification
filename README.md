🐱🐶 Cat-Dog Image Classification

A deep learning project that classifies images as either Cat or Dog using Convolutional Neural Networks (CNN).
This project demonstrates end-to-end image classification, from data preprocessing to model training, evaluation, and prediction.

📌 Features

Classifies images into Cat or Dog categories.

Built using TensorFlow/Keras CNN architecture.

Includes data preprocessing (resizing, normalization, augmentation).

Achieves high accuracy on the test dataset.

Supports custom image prediction.

🛠️ Technologies Used

Python 3

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

OpenCV (for image handling)

📂 Project Structure
Cat-Dog-Classification/
│── data/                     # Dataset (train/test images)
│── models/                   # Saved trained model
│── notebooks/                # Jupyter notebooks for training & testing
│── src/                      # Python scripts for model & preprocessing
│── requirements.txt          # Project dependencies
│── app.py                    # Prediction script
│── README.md                 # Project documentation

📊 Model Architecture

Input Layer → Image size (e.g., 128x128x3)

Conv2D + MaxPooling layers (multiple)

Flatten Layer

Fully Connected Dense Layers

Output Layer → 1 neuron with sigmoid activation
