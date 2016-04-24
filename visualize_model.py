import sys
import theano
import numpy as np
import models
import matplotlib.pyplot as plt
import pickle
import pdb


model_name = "model_tstill_bs1_sl10.p"
N = 12

from load_mnist import load_mnist, pad_mnist, movie_mnist, batch_pad_mnist


def load_model(name):
	return pickle.load(open(name, 'r'))

def visualize_glimpse(image, x, y, delta, sigma, correct):
	result = np.expand_dims(image, axis=2).repeat(3, axis=2)
	(max_x, max_y) = image.shape
	min_x = min_y = 0
	
	channel = 1 if correct else 0
	x = int(x)
	y = int(y)
	delta = int(delta)
	sigma = int(sigma*3)
	right_x = x + delta*(N/2)
	left_x = x - delta*(N/2)
	top_y = y - delta*(N/2)
	bot_y = y + delta*(N/2)

	right_x = right_x if right_x<max_x else max_x
	left_x = left_x if left_x > min_x else min_x
	top_y = top_y if top_y > min_y else min_y
	bot_y = bot_y if bot_y < max_y else max_y
	padding = sigma

	for fx in range(result.shape[1]):
		for fy in range(result.shape[0]):
			color = False
			if fx > left_x-padding and fx < right_x+padding and fy > top_y - padding and fy<bot_y+padding:
				color = True
			if fx > left_x+padding and fx < right_x-padding and fy > top_y + padding and fy<bot_y-padding:
				color = False
			if color:
				result[fy, fx, channel] = 1
	return result

def plot_glimpse_vis(frames, glimpses):
	plt.axis('off')
	a = len(frames)
	for i in range(a):
		frame = frames[i]
		glimpse = glimpses[i]

		plt.subplot(2, a, i+1)
		plt.imshow(frame)
		plt.axis('off')
		plt.subplot(2, a, a+i+1)
		plt.imshow(glimpse, cmap='gray')
		plt.axis('off')
	
	plt.show()

def test_image(image, label):
	x = y = np.asarray(36)
	im = pad_mnist(image, x=x, y=y)[0]

	model = load_model(model_name)
	glimpses = []
	frames = []
	reset = True
	for i in range(4):
		prediction, attention = model.predict(im, reset=reset)
		reset = False
		correct = prediction.argmax() == label

		glimpse = attention[0].reshape((N,N))
		glimpses.append(glimpse)
		frame = visualize_glimpse(im, attention[1], attention[2], attention[3], attention[4], correct=correct)
		frames.append(frame)
	plot_glimpse_vis(frames, glimpses)

if __name__ == '__main__':
	images, labels = load_mnist(dataset="testing", path="mnist")
	images = images/255.

	test_image(images[0], labels[0])