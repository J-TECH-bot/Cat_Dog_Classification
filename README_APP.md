# Cat vs Dog Classification Streamlit App

A beautiful and interactive web application for classifying images of cats and dogs using a trained deep learning model.

## Features

- 🎯 **Real-time Classification**: Upload images and get instant predictions
- 📊 **Confidence Visualization**: See detailed probability scores with visual bars
- 🎨 **Modern UI**: Clean, responsive design with custom styling
- 📱 **User-friendly**: Simple drag-and-drop interface
- 🔍 **Image Processing**: Automatic image preprocessing and resizing

## Prerequisites

Make sure you have the following files in your project directory:
- `model.h5` - Your trained model file
- `app.py` - The Streamlit application
- `requirements.txt` - Dependencies list

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify your model file:**
   Make sure `model.h5` is in the same directory as `app.py`

## Running the App

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   The app will automatically open in your default browser at `http://localhost:8501`

## How to Use

1. **Upload an Image**: Click the file uploader and select an image of a cat or dog
2. **View the Image**: The uploaded image will be displayed
3. **Make Prediction**: Click the "Predict" button to analyze the image
4. **See Results**: View the predicted class, confidence score, and detailed probabilities

## Model Information

- **Input Size**: 150x150 pixels
- **Classes**: Cat, Dog
- **Architecture**: CNN with 4 convolutional layers
- **Training**: Trained on Cats vs Dogs dataset

## File Structure

```
Cat_Dog_Classification/
├── app.py              # Streamlit application
├── model.h5            # Trained model file
├── requirements.txt    # Python dependencies
├── README_APP.md       # This file
└── catvsdog_bharatintern.py  # Original training script
```

## Troubleshooting

- **Model not found**: Ensure `model.h5` is in the same directory as `app.py`
- **Import errors**: Install all dependencies with `pip install -r requirements.txt`
- **Memory issues**: Close other applications if you encounter memory problems

## Customization

You can customize the app by:
- Modifying the CSS styles in the `st.markdown()` section
- Changing the page title and icon
- Adjusting the layout and columns
- Adding more features like batch processing

## Dependencies

- **Streamlit**: Web application framework
- **TensorFlow**: Deep learning framework
- **OpenCV**: Image processing
- **Pillow**: Image handling
- **NumPy**: Numerical computations 