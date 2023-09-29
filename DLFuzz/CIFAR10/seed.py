from tensorflow.keras.datasets import cifar10
from imageio import imwrite
import random
import os
from datetime import datetime

def get_signature():
	now = datetime.now()
	past = datetime(2015, 6, 6, 0, 0, 0, 0)
	timespan = now - past
	time_sig = int(timespan.total_seconds() * 1000)

	return str(time_sig)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
save_dir = './seeds_50_poison/'
if os.path.exists(save_dir):
	for i in os.listdir(save_dir):
		path_file = os.path.join(save_dir, i)
		if os.path.isfile(path_file):
			os.remove(path_file)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)


for i in range(50):
	#x = random.randint(0, len(x_test)-1)

	#save_img = save_dir + str(get_signature()) + '_' + str(y_test[x][0]) + '.png'
	save_img = save_dir + str(i) + '_' + str(y_train[i][0]) + '.png'

    #imwrite(save_img, x_test[x])
	imwrite(save_img, x_train[i])


