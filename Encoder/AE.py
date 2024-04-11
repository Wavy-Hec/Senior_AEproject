# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, losses
from keras.models import Model
import numpy as np

import matplotlib.pyplot as plt

arr = np.load("Encoder/AntObservations.npy")
# delete extra rows
arr = np.delete(arr,slice(27,111),1)

# flatten the array observations
arr  = np.reshape(arr, (-1, 27) )

print(arr.shape)

# shuffle for training
np.random.shuffle(arr)

x_train = arr[:int(arr.shape[0]*.8)]
x_test = arr[int(arr.shape[0]*.8):]

# convert to datatype for model
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train = tf.convert_to_tensor(x_train)
# x_test = tf.convert_to_tensor(x_test)

# Displaying the shapes of the training and testing datasets
print("Shape of the training data:", x_train.shape)
print("Shape of the testing data:", x_test.shape)

# Definition of the Autoencoder model as a subclass of the TensorFlow Model class
class SimpleAutoencoder(Model):
	def __init__(self,latent_dimensions , data_shape):
		super(SimpleAutoencoder, self).__init__()
		self.latent_dimensions = latent_dimensions
		self.data_shape = data_shape

		# Encoder architecture using a Sequential model
		self.encoder = tf.keras.Sequential([
			layers.Flatten(),
			layers.Dense(latent_dimensions, activation='linear'),
		])

		# Decoder architecture using another Sequential model
		self.decoder = tf.keras.Sequential([
			layers.Dense(tf.math.reduce_prod(data_shape), activation='linear'),
			layers.Reshape(data_shape)
		])

	# Forward pass method defining the encoding and decoding steps
	def call(self, input_data):
		encoded_data = self.encoder(input_data)
		decoded_data = self.decoder(encoded_data)
		return decoded_data


# Extracting shape information from the testing dataset
input_data_shape = x_test.shape[1:]

# Specifying the dimensionality of the latent space
latent_dimensions = 26

squeeze_losses = {}

# Correct way to combine range with [1] using list concatenation
for latent_dimensions in list(range(26, 1, -2)) + [1]:
    # Creating an instance of the SimpleAutoencoder model
    simple_autoencoder = SimpleAutoencoder(latent_dimensions, input_data_shape)

    # Compile the model with Adam optimizer and MSE loss function
    simple_autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # Fit the model
    simple_autoencoder.fit(x_train, x_train,
                           epochs=20,
                           shuffle=True,
                           validation_data=(x_test, x_test))

    # Save the model
    simple_autoencoder.save(f'Encoder/AE_{latent_dimensions}', save_format='tf')

    # Get the validation loss of the last epoch
    v = simple_autoencoder.history.history['val_loss'][-1]

    # Store the validation loss in the dictionary with the latent dimension as the key
    squeeze_losses[latent_dimensions] = v

    # Write the latent dimension and its corresponding loss to the file
    with open('Encoder/AE_trained_loss_MSE.txt', 'a') as f:
        f.write(f'{latent_dimensions}: {v}\n')

# Sort the losses by the number of latent dimensions
lists = sorted(squeeze_losses.items())

# Unpack the sorted items for plotting
x, y = zip(*lists)

# Plotting the MSE loss against the number of latent features
plt.plot(x, y)
plt.title('MSE Loss vs. Number of Latent Features')
plt.xlabel('# of features')
plt.ylabel('MSE')
plt.savefig('Encoder/MSE_losses_vs_dimensions')
plt.show()