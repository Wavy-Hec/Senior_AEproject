import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from keras import layers, losses
from keras.models import Model

import gymnasium as gym
from stable_baselines3 import SAC
import torch
import matplotlib.pyplot as plt


env = gym.make("Humanoid-v4", render_mode="human")
model = SAC.load("SAC_2000000.zip", env=env)


obs_arr = np.array(env.reset()[0], dtype=np.double)


obs = env.reset()

#train for longer/get better model
for x in range(5000):
    obs, _, done, _, _ = env.step(env.action_space.sample())
    obs_arr = np.append(obs_arr, obs)
    if done:
      obs = env.reset()
  


obs_arr = np.reshape(obs_arr, (-1, 376) )


loader = torch.utils.data.DataLoader(dataset=obs_arr, batch_size=32, shuffle=True)


import matplotlib.pyplot as plt


np.random.shuffle(obs_arr)


x_train = obs_arr[:int(obs_arr.shape[0]*.8)]
x_test = obs_arr[int(obs_arr.shape[0]*.8):]

# x_train = obs_arr[:int(theRange*.8)]
# x_test = obs_arr[int(theRange*.8):]

print("train and test")
print(x_train.shape, x_test.shape)
# Normalizing pixel values to the range [0, 1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')



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
latent_dimensions = 200


# Creating an instance of the SimpleAutoencoder model
simple_autoencoder = SimpleAutoencoder(latent_dimensions, input_data_shape)


#losses.MeanSquaredError()
simple_autoencoder.compile(optimizer='adam', loss=losses.MeanAbsoluteError())

graph = simple_autoencoder.fit(x_train, x_train,
				epochs=500,
				shuffle=True,
				validation_data=(x_test, x_test))


encoded_imgs = simple_autoencoder.encoder(x_test).numpy()
decoded_imgs = simple_autoencoder.decoder(encoded_imgs).numpy()


import datetime;

plt.plot(graph.history['loss'], label='Training Loss')


plt.title('Autoencoder Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#saving the model
# now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
# newFile = open(f"./savedModels/{now}", 'a')

# for i in encoded_imgs:
# 	newFile.write(str(i))


simple_autoencoder.save(f'savedModels/HAE_990Rwrd_500E_{latent_dimensions}',save_format='tf')





