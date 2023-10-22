import tensorflow as tf

# first we will import the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize to help: improve convergence, reduce sensitivity to learning rate,
# help effective weight sharing (different regions of images),
# avoid vanishing/exploding gradients, and help reduce impact of variations
# in pixel intensity and illumination
x_train, x_test = x_train / 255.0, x_test / 255.0

# Now utilize One-hot encoding in order to perform proper pre processing
# in the correct numerical form for the neural net i.e one -> 1
# Note: y set is 10 categories zero->nine
# ensures that all output categories exist uncomment print if need example
from tensorflow.keras.utils import to_categorical
# print(y_train, y_test)
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
# print(y_train[0], y_test[0])

# If your data doesn't have testing then lets make it!
# now we shoudl build the training and testing sets
# we will take 80% of data for training and 20% of data to test
# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Lets build a CNN model!
model = tf.keras.Sequential() # all of your layers can be a list here

# simplify layers import
import tensorflow.keras.layers as layers

# Lets start with a convolutional input layer
model.add(layers.Conv2D(
    # how many filters should be used?
    # These filter values start random and are learned per iteration
    filters=32,
    # How will the filters be applied to the image?
    # this makes the filter slide over a 3x3 region
    kernel_size=(3, 3),
    # The Rectified Linear Unit (ReLU) activation function is commonly
    # used in CNNs. It introduces non-linearity by setting all negative
    # values to zero and leaving positive values unchanged.
    activation='relu',
    # shape is the dimensions of image and color channels
    input_shape=x_train.shape[1:] + (1,)
))

# Now our convolutional layer will have given us larger
# spatial dimensions of feature maps
# so lets use a method called MapPooling
# MaxPooling is a down-sampling technique used to reduce the spatial dimensions
# of feature maps produced by convolutional layers
# This takes some window and only keeps the maximum value within the window
# default (2, 2)
model.add(layers.MaxPooling2D((2, 2)))

# How stacking covolutional layers can allow the model to define increasingly
# complex and abstract features from input data
model.add(layers.Conv2D(
    filters=64, # Lets try and learn some complex features!
    kernel_size=(3,3),
    activation='relu'
))

# Down-sample
model.add(layers.MaxPooling2D((2, 2)))

# Now before we pass into a fully connected (Dense) layer
# we should flatten into a one-dimensional array
# basically formatting for Dense
model.add(layers.Flatten())

# Now that we have learned some features
# Lets learn how these features are significant

# What is a Dense Layer
# A Dense layer represents a fully connected layer from the previous layer
# This layer performs a **weighted sum** of inputs, adds a **bias term**, and then
# applies an **activation function** to produce an output. The output is a non-linear
# transformation of its input data
# This layer is basically a information processing layer
model.add(layers.Dense(
    # lets define how many neruons this layer has
    units=128,
    activation='relu'
))

# Now lets create the output layer
model.add(layers.Dense(
    # there are 10 classes of output zero->nine
    units=10,
    activation='softmax'
))

# Compile the model
model.compile(
    # The optimizer updates the model's weights during training.
    # Adam is an extended SGD (Stochastic Gradient Descent)
    optimizer='adam',
    # The loss/objective/cost function measures how well the model is
    # performing. Quantifies the difference between predicted vs actual.
    # categorical_crossentropy is for multiclass classification problems
    # categorical cross-entropy loss encourages the model to assign high
    # probabilities to the correct classes while penalizing incorrect
    # class assignments. This makes it a suitable choice for training
    # classification models.
    loss='categorical_crossentropy',
    # There are many metrics, we care about accuracy
    # read about the different ones we can use at tensorflow
    metrics=['accuracy']
)

# Now we have to reshape the data to fit that extra dimension from earlier

# Remeber we added the 1 to tell we are using greyscale 0->255 not color images
# print(x_train.shape)
x_train = x_train.reshape(x_train.shape + (1,))
# print(x_train.shape)
x_test = x_test.reshape(x_test.shape + (1,))

# Train the model!
model.fit(
    # input data
    x=x_train,
    # expected output
    y=y_train,
    # computational efficiency
    batch_size=64,
    # how many times should we fit our data?
    epochs=5,
    # Model is not trained on this
    # Serves as a way to evaluate the model
    # on unseen data it helps YOU understand
    # how well the model is generalizaing
    validation_data=(x_test, y_test)
)

# Evaluate the model
# hover over evaluate to read docs
test_loss, test_accuracy = model.evaluate(
    x=x_test,
    y=y_test,
    verbose=1)
print('Test accuracy: {:.2f}%'.format(test_accuracy * 100))