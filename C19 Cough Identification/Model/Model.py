'''
this is the code for the built CNN model
'''
#1st import library
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

#build CNN model
model = tf.keras.models.Sequential([
    #first_convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #second_convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #third_convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #fourth_convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid') 
])

#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#fit model
model.fit(train_generator, batch_size=32,epochs=100)
model.save("c19.h5")#save_model

#evaluate/test model
accuracy = model.evaluate(validation_generator)
print('\n', 'Test_Accuracy:-', accuracy[1])

#prediction model from validation generator
pred = model.predict(validation_generator)
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(pred, axis=1)
    
#confusion matrix for model peroformance
print('confusion matrix')
print(confusion_matrix(y_true, y_pred))
    
