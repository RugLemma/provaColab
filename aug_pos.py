from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch import config

import Augmentor

p = Augmentor.Pipeline("D:\Download\Progetto di tesi\pycharmP\dl-medicalTrainTestSaveAndLoadTest scorrelato aug Pos\dataset\Pos")
p.random_distortion(0.25, 10, 10, 3)
#p.rotate(probability=0.25, max_left_rotation=15, max_right_rotation=15)

p.rotate90(0.25)
p.rotate180(0.25)
p.rotate270(0.25)

#p.rotate_random_90(0.75)
#p.skew_corner(0.25,0.1)
p.sample(1600)
