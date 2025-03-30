# CNN-Image-Classifier Cat-VS-Dog

This project is a **Convolutional Neural Network (CNN)** implemented using TensorFlow and Keras to classify images as either **cats or dogs**.

## Features
- Uses **Conv2D** layers for feature extraction.
- Applies **MaxPooling** to reduce spatial dimensions.
- Includes **Fully Connected Layers** for classification.
- Uses **Binary Crossentropy** as the loss function.
- Trained using images stored in `dataset/train` and `dataset/test` directories.

## Installation
Ensure you have Python installed along with the necessary libraries:
```bash
pip install tensorflow keras numpy
```

## Dataset Structure
Organize the dataset as follows:
```
dataset/
    train/
        cat/
            cat1.jpg, cat2.jpg, ...
        dog/
            dog1.jpg, dog2.jpg, ...
    test/
        cat/
            cat_test1.jpg, cat_test2.jpg, ...
        dog/
            dog_test1.jpg, dog_test2.jpg, ...
    pred/
        img.png  # Image for prediction
```

## Training the Model
Run the following script to train the model:
```python
model.fit(train_set, steps_per_epoch=20, epochs=10, validation_data=test_set, validation_steps=3)
```

## Making Predictions
To classify a new image:
```python
result = model.predict(test_image)
if result[0][0] >= 0.5:
    print('Dog')
else:
    print('Cat')
```

## Model Architecture
- **Input Layer**: 64x64x3 (RGB image)
- **Conv2D**: 32 filters, (3x3) kernel, ReLU activation
- **Conv2D**: 32 filters, (3x3) kernel, ReLU activation
- **MaxPooling2D**: (2x2) pool size
- **Flatten**
- **Dense**: 128 neurons, ReLU activation
- **Dense**: 1 neuron, Sigmoid activation

## License
This project is open-source and can be modified or extended as needed.


