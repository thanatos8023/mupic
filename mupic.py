import os
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity


class Mupic:
	def __init__(self, att_file, img_dir, sample_size=1000, separator=',', target_size=(640,640)):
		self.target_size = target_size
		self.sample_size = sample_size
		self.model = InceptionV3(include_top=False, weights='imagenet')

		feature_file = 'features/sample_%d.npy' % self.sample_size
		if os.path.exists(feature_file):
			df = pd.read_table(att_file, sep=separator)
			self.attributes = df.head(sample_size)
			print('Already image feature were maden. We will load the feature data from [[ %s ]].' % feature_file)
			self.features = self.load_feature(feature_file)
			print('Successfully load the feature.')
		else:
			self.attributes = self.load_data(att_file, img_dir, separator=separator)
			print('There is no file that already transformed feature map. So we will make feature map now.')
			self.features = self.make_feature_matrix()
			print('Done. Feature map saved on [[ %s ]]' % feature_file)
		


	def load_feature(self, path):
		X = np.load(path)
		return X


	def load_data(self, att_file, img_dir, separator=','):
		# Read attribute table
		atts = pd.read_table(att_file, sep=separator)

		# Sample
		if not self.sample_size == 0:
			atts = atts.head(self.sample_size)

		# Get image files path as list
		img_files = os.listdir(img_dir)

		# Make images path list of samples
		ids = atts['track_id'].values
		paths = [os.path.join(img_dir, p) for p in img_files if os.path.splitext(p)[0] in ids]

		# Add a new column
		atts['image_path'] = paths

		return atts


	def make_feature_matrix(self, save=True):
		X = []
		img_files_path = self.attributes['image_path'].values

		for img_file in tqdm(img_files_path):
			features = self.make_feature(img_file)
			X.append(features)

		X = np.array(X)

		if save:
			np.save('features/sample_%d' % self.sample_size, X)

		return X


	def make_feature(self, img_path):
		img = image.load_img(img_path, target_size=self.target_size)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		features = self.model.predict(x)
		features = features.mean(axis=(0, 3))
		features = features.ravel()

		return features


	def train(self, y_col):
		y = self.attributes[y_col].values

		mlr = LinearRegression()
		mlr.fit(self.features, y)

		return mlr


	def set_input(self, img_path):
		self.input = self.make_feature(img_path)


	def predict(self, model):
		return model.predict([self.input])


	def recommend_music(self, X_vect, y_mat):
		result = cosine_similarity(X_vect, y_mat)

		max_idx = np.where(result == np.amax(result))[1][0]

		artist = self.attributes['artist_name'][max_idx]
		track = self.attributes['track_name'][max_idx]

		return artist, track


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--target', '-t', help='Input image for recommendation')
	parser.add_argument('--feature', '-f', required=False, help='Preprocessed file path')

	args = parser.parse_args()

	attribute_file = 'SpotifyFeatures.csv'
	image_directory = 'images'

	mupic = Mupic(attribute_file, image_directory)

	acoustic_model = mupic.train('acousticness')
	dance_model = mupic.train('danceability')
	energy_model = mupic.train('energy')
	valence_model = mupic.train('valence')

	mupic.set_input(args.target)

	acousticness = mupic.predict(acoustic_model)
	danceability = mupic.predict(dance_model)
	energy = mupic.predict(energy_model)
	valence = mupic.predict(valence_model)

	input_vect = [acousticness, danceability, energy, valence]
	print('Picture attributes:', input_vect)
	input_vect = np.array(input_vect).transpose()
	base_matrix = mupic.attributes[['acousticness', 'danceability', 'energy', 'valence']].to_numpy()

	artist, track = mupic.recommend_music(input_vect, base_matrix)

	print('Artist:', artist)
	print('Track:', track)


if __name__ == '__main__':
	main()