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
#import scipy

#from scipy.misc import pilutil

class FetexImage(object):
	verbose = None
	# width = 128
	# height = 128
	width = 64
	height = 64

	# width = 90
	# height = 90
	"""docstring for FetexImage"""
	def __init__(self,verbose = False):
		super(FetexImage, self).__init__()
		self.verbose = verbose

	def calculate_average_image(self,im_paths):
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
		
		w,h=imlist[0].size
		N=len(imlist)
		arr=np.zeros((h,w,3),theano.config.floatX)
		#image_index = 0
		for im in imlist:
			imarr=np.array(im,dtype=theano.config.floatX)
			#print len(imarr)
			try:
				arr=arr+imarr/N
			except Exception, e:
				print e
				#print image_index
				#print im.filename
			#image_index += 1
			

		arr=np.array(np.round(arr),dtype=np.uint8)
		#arr=np.array(np.round(arr),dtype=theano.config.floatX)
		average_image=Image.fromarray(arr,mode="RGB")
		#average_image.show()
		
		return imlist,average_image

	def substract_average_image(self, im, average_image):
		#arr=np.array(np.round(arr),dtype=np.uint8)
		im_minus_avg = np.array(np.round(im),dtype=np.uint8) - np.array(np.round(average_image),dtype=np.uint8)
		#im_minus_avg = np.array(np.round(im),dtype='int64') - np.array(np.round(average_image),dtype='int64')
		#im_minus_avg[im_minus_avg < 0] = 0
		#im_minus_avg = abs(im_minus_avg)
		#print im_minus_avg
		#im_minus_avg = np.array(im, dtype=theano.config.floatX) - np.array(average_image, dtype=theano.config.floatX)
		#arr=np.array(np.round(im_minus_avg),dtype=np.uint8)
		#arr=np.array(np.round(im_minus_avg),dtype=theano.config.floatX)
		#im_minus_avg=np.array(np.round(im_minus_avg),dtype=theano.config.floatX)
		im_minus_avg=Image.fromarray(im_minus_avg,mode="RGB")
		
		return im_minus_avg

	def scale_and_crop_img(self,img):
		#img = Image.open(im_path)

		# size = 256, 256
		# img.thumbnail(size, Image.ANTIALIAS)
		# img.save('/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/n03131574-craddle/n03131574_16-res.JPEG', "JPEG")
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


	def add_bg_square(self,img,r,b,g):
		"return a background-color image having the img in exact center"
		size = (max(img.size),)*2
		layer = Image.new('RGB', size, (r,b,g))
		layer.paste(img, tuple(map(lambda x:(x[0]-x[1])/2, zip(size, img.size))))
		return layer

	def average_image_color(self,filename):
		i = Image.open(filename)
		h = i.histogram()
	
		# split into red, green, blue
		r = h[0:256]
		g = h[256:256*2]
		b = h[256*2: 256*3]
	
		# perform the weighted average of each channel:
		# the *index* is the channel value, and the *value* is its weight
		return (
			sum( i*w for i, w in enumerate(r) ) / sum(r),
			sum( i*w for i, w in enumerate(g) ) / sum(g),
			sum( i*w for i, w in enumerate(b) ) / sum(b)
		)
	
	def convert_to_bw_and_scale(self):
		im_path = '/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/n03131574-craddle-resized/n03131574_10027.JPEG'
		im = Image.open(im_path)
		
		# Get monochrome pixels
		im_aux = im.convert('L')
		pixels_monochrome = np.array(list(im_aux.getdata()), dtype='float32')
		
		# scale between 0-1 to speed up computations
		# print type(pixels_monochrome)
		min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
		pixels_monochrome = min_max_scaler.fit_transform(pixels_monochrome)

		return pixels_monochrome


		# im.convert('1').getdata() # Convert to BW

		pixels = im.load() # this is not a list, nor is it list()'able
		width, height = im.size
	
		all_pixels = []
		for x in range(width):
			for y in range(height):
				# Append pixels to all_pixels list providing this is RGB
				# cpixel = pixels[x, y]
				# all_pixels.append(cpixel)
				
				# Convert to monochrome(only one value)
				cpixel = pixels[x, y]
				bw_value = int(round(sum(cpixel) / float(len(cpixel))))
				# the above could probably be bw_value = sum(cpixel)/len(cpixel)
				all_pixels.append(bw_value)

				# Or to get the luminance (weighted average):
				# cpixel = pixels[x, y]
				# luma = (0.3 * cpixel[0]) + (0.59 * cpixel[1]) + (0.11 * cpixel[2])
				# all_pixels.append(luma)

				# Or pure 1-bit looking black and white:
				# cpixel = pixels[x, y]
				# if round(sum(cpixel)) / float(len(cpixel)) > 127:
    # 				all_pixels.append(255)
				# else:
    # 				all_pixels.append(0)

		#print all_pixels
		# print len(all_pixels)
		# all_pixels.show()


	def processImagesPipeline(self,folder,support_per_class = None):

		X = []
		Y = []
		lb = preprocessing.LabelBinarizer()
		#lb.fit_transform(['n03131574-craddle','n04222210-single-bed'])
		#classes = ['n07730207-carrot','n04222210-single-bed', 'n00015388-animal','n00017222-flower','n00523513-sport','n01503061-bird','n12992868-fungus']
		#classes = ['n07730207-carrot','n04222210-single-bed', 'n00015388-animal','n00017222-flower','n00523513-sport','n01503061-bird','n12992868-fungus']
		#classes = ['n07730207-carrot','n04222210-single-bed', 'n00015388-animal','n00017222-flower','n00523513-sport','n01503061-bird','n12992868-fungus']
		classes = ["windows-accessories","doors-and-doorways","rooflights-roof-windows","building-areas-systems","ground-substructure","floors-accessories","stairs","ceilings","wall-finishes","roof-structures-finishes","hvac-cooling-systems","insulation","lighting","furniture-fittings","external-works","building-materials","bathroom-sanitary-fittings","structural-frames-walls","drainage-water-supply","communications-transport-security","green-building-products"]
		#classes = ['n07730207-carrot','n04222210-single-bed']
		lb.fit_transform(classes)

		im_paths = []
		im_labels = []
		#for image_type in ['n03131574-craddle','n04222210-single-bed']:
		
		for image_type in classes:

			mypath = folder + image_type
			onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
			class_support = 0
			for file_name in onlyfiles:
				if file_name != '.DS_Store':
					im_path = mypath = folder + image_type + '/' + file_name
					#outfile = folder + image_type + '-resized/' + file_name
					im_paths.append(im_path)
					im_labels.append(image_type)
				class_support += 1
				if support_per_class != None and class_support ==  support_per_class:
					break

		combined = zip(im_paths, im_labels)
		random.shuffle(combined)

		im_paths[:], im_labels[:] = zip(*combined)


		print "load, scale, crop and calculate image average"
		imlist,average_image = self.calculate_average_image(im_paths)
		#average_image.show()
		return imlist

		i = 0
		#for im_path in im_paths:
		for im in imlist:
			# Substract average image and convert it to BW
			# Then convert it to monochrome and append it to X
			# Append the label to Y
			try:
				#im_raw = np.array(im,dtype=theano.config.floatX)
				#im_raw = np.asarray(im, dtype=theano.config.floatX) / 256.
				#print im_raw
				#im.show()
				im_ini = im
				#print im.size
				im = self.substract_average_image(im, average_image)
				#print im.size
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
				# im_r = np.asarray(im_r, dtype=theano.config.floatX) / 256.
				# im_r_2 = np.asarray(im_r_2, dtype=theano.config.floatX) / 256.
				# im_r_3 = np.asarray(im_r_3, dtype=theano.config.floatX) / 256.
				#im_f = np.asarray(im_f, dtype=theano.config.floatX) / 256.
				
				#im = im.transpose(2, 0, 1)
				#X.append(np.array(im, dtype=theano.config.floatX))
				#X.append(np.array(im_raw, dtype=theano.config.floatX))
				X.append(im)
				# X.append(im_r)
				# X.append(im_r_2)
				# X.append(im_r_3)
				#X.append(im_f)
				# X.append(im_f_2)
				# X.append(im_f_3)
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
				# Y.append(label)	
				# Y.append(label)	
			except Exception, e:
				raise e


			#break
			
			# print Y
			# print len(Y)
			#im_aux.show()
			
			# if self.verbose:
			# 	#sys.stdout.write("\r Image Type: {0} File Name: {1} Process: {2}/{3}".format(image_type, file_name, count, len(im_paths)))
			# 	#sys.stdout.write("\r File Name: {0} Process: {1}/{2}".format(im_path, count, len(im_paths)))
			# 	sys.stdout.write("\r Process: {0}/{1}".format(i, len(im_paths)))
			# 	sys.stdout.flush()

			# if i == 2:
			# 	break

			i += 1

		train_length = int(round(len(X) * 0.60))
		valid_length = int(round(len(X) * 0.20))
		test_length = int(round(len(X) * 0.20))

		def flaten_aux(V):
			return V.flatten(order='F')

		X = map(flaten_aux, X)

		X_train = X[0:train_length]
		X_valid = X[train_length: (train_length + valid_length)]
		X_test = X[-test_length:]

		# def flaten_aux(V):
		# 	#return V.flatten(order='C')
		# 	return V.flatten(order='F')
		# #vfunc = np.vectorize(flaten_aux)

		# X_train = map(flaten_aux, X_train)
		# X_valid = map(flaten_aux, X_valid)
		# X_test = map(flaten_aux, X_test)
		#print X_train
		#print X_train.size
		#X_train = vfunc(X_train)
		# X_valid = vfunc(X_train)X_valid_flat.flatten(order='F')
		# X_test = X_test_flat.flatten(order='F')


		#X_train = X_train.flatten(order='F')
		#X_valid = X_valid.flatten(order='F')
		#X_test = X_test.flatten(order='F')

		# Y_train = np.array(Y[0:train_length], dtype='int32')
		# Y_valid = np.array(Y[train_length:(train_length + valid_length)], dtype='int32')
		# Y_test = np.array(Y[-test_length:], dtype='int32')
		# Y = np.array(Y, dtype='int32')
		# Y = Y.reshape(Y.shape[0], -1) 
		# print Y
		Y_train = Y[0:train_length]
		Y_valid = Y[train_length:(train_length + valid_length)]
		Y_test = Y[-test_length:]
		#print Y_test

		train_set = [X_train,Y_train]
		valid_set = [X_valid,Y_valid]
		test_set = [X_test,Y_test]

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
ORDER BY total DESC
		"""

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
	fe = FetexImage(verbose=True)
	#fe.scale_and_crop_test('/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/n03131574-craddle/n03131574_16.JPEG')
	#print fe.convert_to_bw_and_scale()
	train_set,valid_set,test_set = fe.processImagesPipeline(folder,support_per_class=10)
	#fe.createFolderStructure()

