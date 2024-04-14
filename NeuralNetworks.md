# Neural Networks
Neural networks are computational systems inspired by the structure and functioning of the human brain. They are the fundamental component of many Machine learning models. Neural networks learn by example.

Neural networks are made of layers of neurons. These neurons form the core processing units of the network. The data is passed to the network through the input layer. The output layer predicts the final outcome. The layers in between input and out layers, called, the hidden layers performs most of the computations.

![image](https://github.com/vvvvvvss/Neural-networks/assets/148562671/4589325c-52f1-415f-af9f-46d98bb3f5f6)

Neural Network architecture
Neural network come in different types:

#### Convolutional Neural Networks (CNN)   
#### Artificial Neural Networks (ANN)    
#### Recurrent Neural Networks (RNN)   

# Convolutional Neural Networks
CNNs are special kind of neural networks that are used for processing data that has a grid like topology. CNN was built on the inspiration of the visual cortex, a part of the human brain. In CNN, we use convolution operation. A neural network can be identified as CNN if there is atleast one convolution layer. Here is the architecture of a CNN:

#### Convolution layer  
#### Pooling layer   
#### Full-Connected layer   
An important application of CNNs are in image classification.

![image](https://github.com/vvvvvvss/Neural-networks/assets/148562671/d86b5a83-5970-4106-a7f2-5711195d7517)    

In the RGB model, the colour image is actually composed of three such matrices corresponding to three colour channels — red, green and blue. In black-and-white images we only need one matrix. Each of these matrices stores values from 0 to 255.

It is a process where we take a small matrix of numbers (called kernel or filter), we pass it over our image and transform it based on the values from filter.

`import tensorflow as tf  
from tensorflow.keras import layers, models  
 #Define the CNN model   
model = models.Sequential()    
#Add convolutional layers   
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))   
model.add(layers.MaxPooling2D((2, 2)))     
model.add(layers.Conv2D(64, (3, 3), activation='relu'))   
model.add(layers.MaxPooling2D((2, 2)))    
model.add(layers.Conv2D(128, (3, 3), activation='relu'))  
model.add(layers.MaxPooling2D((2, 2)))    
model.add(layers.Conv2D(128, (3, 3), activation='relu'))  
model.add(layers.MaxPooling2D((2, 2)))  
#Flatten the output for dense layers   
model.add(layers.Flatten())   
#Add dense layers   
model.add(layers.Dense(512, activation='relu'))    
model.add(layers.Dense(1, activation='sigmoid'))    
#Compile the model   
model.compile(optimizer='adam',    
              loss='binary_crossentropy',    
              metrics=['accuracy'])       
#Display the model summary   
model.summary()`   
# Artificial Neural Networks (ANN)   
Artificial Neural Networks contain artificial neurons which are called units. These units are arranged in a series of layers that together constitute the whole Artificial Neural Network in a system.

Similar to CNNs, Artificial Neural Network has an input layer, an output layer as well as hidden layers. The input layer receives data from the outside world which the neural network needs to analyze or learn about. Then this data passes through one or multiple hidden layers that transform the input into data that is valuable for the output layer. Finally, the output layer provides an output in the form of a response of the Artificial Neural Networks to input data provided.

Artificial Neural Networks (ANNs) have found a wide range of applications across various domains due to their ability to learn complex patterns and relationships from data. Image Recognition and Classification, Speech Recognition and Autonomous Vehicles are a few applications of ANNs.

` import tensorflow as tf
from tensorflow.keras import layers, models
#Define the speech recognition model
model = models.Sequential()
#Add LSTM layer
model.add(layers.LSTM(128, input_shape=(None, 13)))  
#13 features extracted from the audio, input shape is (time_steps, features)
#Add dense layer for classification
model.add(layers.Dense(num_classes, activation='softmax'))  
#num_classes is the number of output classes
#Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#Display the model summary
model.summary()`
