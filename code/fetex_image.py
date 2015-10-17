from PIL import Image
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import random
from sklearn import preprocessing
import cPickle
import theano
from PIL import ImageFilter
import os

class FetexImage(object):
	verbose = None
	width = 64
	height = 64
	support_per_class = None
	folder = None

	"""docstring for FetexImage"""
	def __init__(self,verbose = False,support_per_class = None, folder = None):
		super(FetexImage, self).__init__()
		self.verbose = verbose
		self.support_per_class = support_per_class
		self.folder = folder

	def load_images(self,im_paths):

		"""Loads all the images paths into a PIL Image object. If an image is in RGBA, then it converts it into a RGB. 
		It return a list of PIL Image Objects"""

		imlist = []
		j = 0
		for im_path in im_paths:
			im = None
			
			try:
				im = Image.open(im_path)
			except Exception, e:
				print e
			
			if im != None:
				im_aux = np.array(im,dtype=theano.config.floatX)
				
				try:
					if im_aux.shape[2] == 4:
						background = Image.new("RGB", im.size, (255, 255, 255))
						background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
						im = background
						im_aux = np.array(background,dtype=theano.config.floatX)
				except Exception, e:
					print e
				
				try:
					if im_aux.shape[2] == 3:
						im_aux = self.scale_and_crop_img(im)
						imlist.append(im_aux)
					else:
						print "invalid image: {} size:{}".format(im.filename, im_aux.shape)
	
				except Exception, e:
					print e
	
			if self.verbose:
				sys.stdout.write("\r Process: {0}/{1}".format(j, len(im_paths)))
				sys.stdout.flush()

			j += 1
			# if j == 30:
			# 	break
		return imlist

	def calculate_average_image(self,imlist):
		"""Calculates the average image given PIL Image list"""

		w,h=imlist[0].size
		N=len(imlist)
		arr=np.zeros((h,w,3),theano.config.floatX)
		for im in imlist:
			imarr=np.array(im,dtype=theano.config.floatX)
			try:
				arr=arr+imarr/N
			except Exception, e:
				print e
			
		arr=np.array(np.round(arr),dtype=np.uint8)
		#arr=np.array(np.round(arr),dtype=theano.config.floatX)
		average_image=Image.fromarray(arr,mode="RGB")

		return average_image

	def substract_average_image(self, im, average_image):
		""" Normalise an image by substracting the average image of the dataset. """
		im_minus_avg = np.array(np.round(im),dtype=np.uint8) - np.array(np.round(average_image),dtype=np.uint8)
		im_minus_avg=Image.fromarray(im_minus_avg,mode="RGB")
		return im_minus_avg

	def scale_and_crop_img(self,img):
		""" Scale the image to width and height. If the image ratio is not 1:1, then we need the smaller side to be width/height and then crop the center of the image.
		You can uncomment the commented code to transfor the images to Black and White"""

		if img.size[0] < img.size[1]:
			basewidth = self.width
			wpercent = (basewidth/float(img.size[0]))
			hsize = int((float(img.size[1])*float(wpercent)))
			#img = img.resize((basewidth,hsize), Image.ANTIALIAS).convert('L')
			img = img.resize((basewidth,hsize), Image.ANTIALIAS)

		else:
			baseheight = self.height
			hpercent = (baseheight/float(img.size[1]))
			wsize = int((float(img.size[0])*float(hpercent)))
			#img = img.resize((wsize,baseheight), Image.ANTIALIAS).convert('L')
			img = img.resize((wsize,baseheight), Image.ANTIALIAS)

		half_the_width = img.size[0] / 2
		half_the_height = img.size[1] / 2
		img = img.crop(
    	(
	        half_the_width - (self.width / 2),
        	half_the_height - (self.height / 2),
        	half_the_width + (self.width / 2),
        	half_the_height + (self.height / 2)
    	)
		)

		return img
	
	def load_paths_and_labels(self,classes):
		""" The classes parameter has the names of all the classes the images belong to. 
		Each class name is also the name of the folder where the images for that class are stored. 
		Therefore this function loads all the image paths and their class into lists and returns them."""
		im_paths , im_labels = [], [] 

		for image_type in classes:
			mypath = self.folder + image_type
			onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
			class_support = 0
			for file_name in onlyfiles:
				if file_name != '.DS_Store':
					im_path = mypath = self.folder + image_type + '/' + file_name
					im_paths.append(im_path)
					im_labels.append(image_type)
				class_support += 1
				if self.support_per_class != None and class_support ==  self.support_per_class:
					break

		combined = zip(im_paths, im_labels)
		random.shuffle(combined)

		im_paths[:], im_labels[:] = zip(*combined)
		
		return im_paths,im_labels

	def data_augmentation_and_vectorization(self,imlist, average_image,lb,im_labels):
		""" This function applies data augmentation to the images in order to reduce overfitting. 
		Potential transformations: rotation, filtering and normalisation (Norm is not data autmentation). 
		Then it converts the images into numpy arrays and binarizes the labels into numeric classes. 
		Finally we store the arrays and labels into X and Y and return them."""
		X,X_original,Y = [] ,[] , []

		i = 0
		for im in imlist:
			try:
				im_ini = im
				im_original = np.asarray(im, dtype=theano.config.floatX) / 256.
				im = self.substract_average_image(im, average_image)
				#im.show()
				#Rotations 
				# im_r = im.rotate(15)
				# im_r_2 = im.rotate(-15)
				# im_r_3 = im.rotate(180)
				#im_r.show()
				#im_r_2.show()

				#Filters
				#im_f = im_ini.filter(ImageFilter.DETAIL)
				
				im = np.asarray(im, dtype=theano.config.floatX) / 256.
				#self.reconstructImage(im).show()

				# im_r = np.asarray(im_r, dtype=theano.config.floatX) / 256.
				# im_r_2 = np.asarray(im_r_2, dtype=theano.config.floatX) / 256.
				# im_r_3 = np.asarray(im_r_3, dtype=theano.config.floatX) / 256.
				#im_f = np.asarray(im_f, dtype=theano.config.floatX) / 256.
				
				#im = im.transpose(2, 0, 1)
				#X.append(np.array(im, dtype=theano.config.floatX))
				#X.append(np.array(im_raw, dtype=theano.config.floatX))
				X.append(im)
				X_original.append(im_original)
				# X.append(im_r)
				# X.append(im_r_2)
				# X.append(im_r_3)
				#X.append(im_f)

				# X.append(np.array(im_r, dtype=theano.config.floatX))
				# X.append(np.array(im_r_2, dtype=theano.config.floatX))

				#Uncomment this if you want to work with monochrome
				# im = im.convert('L')
				# pixels_monochrome = np.array(list(im.getdata()), dtype=np.float)
							
				# # scale between 0-1 to speed up computations
				# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=True)
				# pixels_monochrome = min_max_scaler.fit_transform(pixels_monochrome)

				# X.append(pixels_monochrome)

				#Y.append(lb.transform([im_labels[i]])[0][0])
				label = lb.transform([im_labels[i]])[0][0]
				# label = np.asarray(label, dtype=theano.config.floatX)
				# label = np.ndarray.astype(label,dtype=theano.config.floatX) 
				Y.append(label)
				# Y.append(label)
				# Y.append(label)	
				#Y.append(label)	
				# Y.append(label)	
			except Exception, e:
				raise e

			# if i == 5:
			# 	break

			i += 1
			if self.verbose:
				sys.stdout.write("\r Process: {0}/{1}".format(i, len(imlist)))
				sys.stdout.flush()

		output = open('../data/X_original.pkl', 'wb')
		cPickle.dump(X_original, output,protocol=-1)
		output.close()

		return X,Y

	def binarize_classes(self):
		lb = preprocessing.LabelBinarizer()
		#classes = ['n07730207-carrot','n04222210-single-bed', 'n00015388-animal','n00017222-flower','n00523513-sport','n01503061-bird','n12992868-fungus']
		classes = ["windows-accessories","doors-and-doorways","rooflights-roof-windows","building-areas-systems","ground-substructure","floors-accessories","stairs","ceilings","wall-finishes","roof-structures-finishes","hvac-cooling-systems","insulation","lighting","furniture-fittings","external-works","building-materials","bathroom-sanitary-fittings","structural-frames-walls","drainage-water-supply","communications-transport-security","green-building-products"]
		
		if self.verbose:
			print "Num classes:{}".format(len(classes))

		lb.fit_transform(classes)
		return classes,lb
	
	def flaten_aux(self,V):
		return V.flatten(order='F')

	def create_train_validate_test_sets(self,X,Y):
		""" Create Train, validation and test set"""
		train_length = int(round(len(X) * 0.60))
		valid_length = int(round(len(X) * 0.20))
		test_length = int(round(len(X) * 0.20))

		X_train = X[0:train_length]
		X_valid = X[train_length: (train_length + valid_length)]
		X_test = X[-test_length:]

		X_train = map(self.flaten_aux, X_train)
		X_valid = map(self.flaten_aux, X_valid)
		X_test = map(self.flaten_aux, X_test)

		#self.reconstructImage(X_train[0]).show()

		#X = map(self.flaten_aux, X)

		# X_train = X[0:train_length]
		# X_valid = X[train_length: (train_length + valid_length)]
		# X_test = X[-test_length:]

		Y_train = Y[0:train_length]
		Y_valid = Y[train_length:(train_length + valid_length)]
		Y_test = Y[-test_length:]

		train_set = [X_train,Y_train]
		valid_set = [X_valid,Y_valid]
		test_set = [X_test,Y_test]

		if self.verbose:
			print "X_train {} X_validation {} X_test {}".format(len(X_train),len(X_valid),len(X_test))
			print "Y_train {} Y_validation {} Y_test {}".format(len(Y_train),len(Y_valid),len(Y_test))

		output = open('../data/train_set.pkl', 'wb')
		cPickle.dump(train_set, output,protocol=-1)
		output.close()

		output = open('../data/valid_set.pkl', 'wb')
		cPickle.dump(valid_set, output,protocol=-1)
		output.close()

		output = open('../data/test_set.pkl', 'wb')
		cPickle.dump(test_set, output,protocol=-1)
		output.close()
		
		return train_set,valid_set,test_set

	def reconstructImage(self,arr):
		
		""" Reconstruct an image from array """
		arr = arr * 256
		#arr = np.array(np.round(arr),dtype=np.uint8)
		arr = np.array(arr,dtype=np.uint8)

		# We need to transpose the array because we flatten X by columns
		#arr = arr.T
		a = arr.reshape((self.width, self.height,3))

		im = Image.fromarray(a,mode="RGB")

		return im

	def processImagesPipeline(self):
		""" This is the main pipeline to process the images"""

		if self.verbose:
			print "...binarize_classes"

		classes, lb = self.binarize_classes()

		if self.verbose:
			print "...load_paths_and_labels"

		im_paths, im_labels = self.load_paths_and_labels(classes)

		if self.verbose:
			print "...load_images"

		imlist = self.load_images(im_paths)

		if self.verbose:
			print "...calculate_average_image"

		average_image = self.calculate_average_image(imlist)

		if self.verbose:
			print "\n...data_augmentation_and_vectorization\n"
		
		X,Y = self.data_augmentation_and_vectorization(imlist,average_image,lb,im_labels)
		
		if self.verbose:
			print "\n...create_train_validate_test_sets\n"

		train_set,valid_set,test_set =  self.create_train_validate_test_sets(X, Y)

		return train_set,valid_set,test_set

	def createFolderStructure(self):

		""" query to get the number of images per category
				SELECT categories.id,categories.slug, categories.order
		FROM categories
		WHERE categories.order = 10
		
				SELECT COUNT(*) as total, products.type, categories.`name`
		FROM documents 
		JOIN products ON documents.`product_id` = products.id
		JOIN `categories` ON categories.id = products.`category_id`
		WHERE documents.type = 'photo'
		AND products.`category_id` IS NOT NULL
		GROUP BY products.`category_id`
		ORDER BY total DESC"""

		with open(self.path + 'data/categories.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			next(reader, None)  # skip the headers
			for row in reader:
				directory = self.path + 'data/categories/' + str(row[1])
				if not os.path.exists(directory):
					os.makedirs(directory)
	def downloadImages(self):

		""" query to get all the images and slug
				SELECT documents.`name`, documents.`url`, categories.`slug`
		FROM documents 
		JOIN products ON documents.`product_id` = products.id
		JOIN `categories` ON categories.id = products.`category_id`
		WHERE documents.type = 'photo'
		AND products.`category_id` IS NOT NULL
		"""
		with open(self.path + 'data/images.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			next(reader, None)  # skip the headers
			i = 0
			for row in reader:
				print "i:{} url:{}".format(i,row[1])
				
				file_path = self.path + "data/categories/" + row[2] + '/' + row[0]
				if not os.path.exists(file_path):
					urllib.urlretrieve(row[1], file_path )

				i += 1


if __name__ == '__main__':
	
	folder = '/Applications/MAMP/htdocs/DeepLearningTutorials/data/categories/'
	#folder = '/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/'
	#fe = FetexImage(verbose=True,support_per_class=10,folder=folder)
	fe = FetexImage(verbose=True,support_per_class=1,folder=folder)
	#fe.scale_and_crop_test('/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/n03131574-craddle/n03131574_16.JPEG')
	#print fe.convert_to_bw_and_scale()
	train_set,valid_set,test_set = fe.processImagesPipeline()
	#fe.createFolderStructure()

