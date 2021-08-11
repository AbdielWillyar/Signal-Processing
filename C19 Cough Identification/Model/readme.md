This research begins by preprocessing the data, where the voice data is converted to a spectrogram image by only taking 1 second of sound from each existing data. After that, the image data from the spectrogram that has been obtained is preprocessed again using ImageDataGenerator in the Keras library, by rescale and setting the target size.
The data formed is divided into training and test data and entered into the CNN model, with the CNN architecture:

* Conv2D layer – add 4 convolutional (16 filters, 32 filters, 64 filters, size of 3*3, and ReLU as activation function)
* Max Pooling – MaxPool2D with 2*2 layers
* Flatten layer to squeeze the layers into 1 dimension
* Dropout Layer(0.5)
* Dense, feed-forward neural network(256 nodes with ReLU as activation function
* 2 output layers with Sigmoid as activation function
* loss = binary_crossentropy
* optimizer = adam

After being tested, this model has a loss of 0.3084 and an accuracy of 0.9130

This confusion matrix model can be seen in the confusion matrix image (CMc19.png) with the label:
0 : Negative
1: Positive
