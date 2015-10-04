from PIL import Image
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import random
from sklearn import preprocessing
import cPickle
#import scipy
#from scipy.misc import pilutil

class FetexImage(object):
	verbose = None
	"""docstring for FetexImage"""
	def __init__(self,verbose = False):
		super(FetexImage, self).__init__()
		self.verbose = verbose

	def scale_and_crop_img(self,img,convert_to_L = True,average_image = None):
		#img = Image.open(im_path)

		# size = 256, 256
		# img.thumbnail(size, Image.ANTIALIAS)
		# img.save('/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/n03131574-craddle/n03131574_16-res.JPEG', "JPEG")
		
		if img.size[0] < img.size[1]:
			basewidth = 256
			wpercent = (basewidth/float(img.size[0]))
			hsize = int((float(img.size[1])*float(wpercent)))
			#img = img.resize((basewidth,hsize), Image.ANTIALIAS).convert('L')
			img = img.resize((basewidth,hsize), Image.ANTIALIAS)

		else:
			baseheight = 256
			hpercent = (baseheight/float(img.size[1]))
			wsize = int((float(img.size[0])*float(hpercent)))
			#img = img.resize((wsize,baseheight), Image.ANTIALIAS).convert('L')
			img = img.resize((wsize,baseheight), Image.ANTIALIAS)

		half_the_width = img.size[0] / 2
		half_the_height = img.size[1] / 2
		img = img.crop(
    	(
	        half_the_width - 128,
        	half_the_height - 128,
        	half_the_width + 128,
        	half_the_height + 128
    	)
		)

		# Substract the average image from this image
		if average_image != None:
			im_minus_avg = np.array(img, dtype=np.float) - np.array(average_image, dtype=np.float)
			arr=np.array(np.round(im_minus_avg),dtype=np.uint8)
			im_minus_avg=Image.fromarray(arr,mode="RGB")
			img = im_minus_avg

		if convert_to_L:
			img = img.convert('L')

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


	def processImagesPipeline(self,folder):

		X = []
		Y = []
		lb = preprocessing.LabelBinarizer()
		#lb.fit_transform(['n03131574-craddle','n04222210-single-bed'])
		lb.fit_transform(['n07730207-carrot','n04222210-single-bed'])

		im_paths = []
		im_labels = []
		#for image_type in ['n03131574-craddle','n04222210-single-bed']:
		for image_type in ['n07730207-carrot','n04222210-single-bed']:
			mypath = folder + image_type
			onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
			for file_name in onlyfiles:
				if file_name != '.DS_Store':
					im_path = mypath = folder + image_type + '/' + file_name
					#outfile = folder + image_type + '-resized/' + file_name
					im_paths.append(im_path)
					im_labels.append(image_type)

		combined = zip(im_paths, im_labels)
		random.shuffle(combined)

		im_paths[:], im_labels[:] = zip(*combined)

		print "calculating image average"
		imlist = []
		j = 0
		for im_path in im_paths:

			im = Image.open(im_path)
			im_aux = self.scale_and_crop_img(im,False,None)
			imlist.append(im_aux)

			if self.verbose:
				sys.stdout.write("\r Process: {0}/{1}".format(j, len(im_paths)))
				sys.stdout.flush()

			j += 1
			# if j == 10:
			# 	break
		
		w,h=imlist[0].size
		N=len(imlist)
		arr=np.zeros((h,w,3),np.float)
		
		for im in imlist:
			imarr=np.array(im,dtype=np.float)
			#print len(imarr)
			try:
				arr=arr+imarr/N
			except Exception, e:
				print e
			

		arr=np.array(np.round(arr),dtype=np.uint8)
		average_image=Image.fromarray(arr,mode="RGB")
		average_image.show()
		

		# aux = np.array(imlist[0], dtype=np.float) - np.array(out, dtype=np.float)
		# arr=np.array(np.round(aux),dtype=np.uint8)
		# out=Image.fromarray(arr,mode="RGB")
		# out.show()

		#print R
		#return None
#		shuffle(im_paths)

		i = 0
		#train_set,validation_set,test_set = [],[],[]
		#for im_path in im_paths:
		for im in imlist:
			#Get average color so if the aspect ratio is greater or smaller that 1 we fill the blank with the average color
			#avg_color = self.average_image_color(im_path)
			
			#Add background color (if needed) and resize
			im = Image.open(im_path)
			#square_one = self.add_bg_square(im,avg_color[0],avg_color[1],avg_color[2])
			#Add white background
			#square_one = self.add_bg_square(im,0,0,0) OLD
			#square_one.resize((256, 256), Image.ANTIALIAS).save(outfile) # uncomment this if you want to save the image
			
			#Resize and convert to BW
			#im_aux = square_one.resize((256, 256), Image.ANTIALIAS).convert('L') OLD
			im_aux = self.scale_and_crop_img(im,True,average_image)
			#im_aux.show()
			pixels_monochrome = np.array(list(im_aux.getdata()), dtype='float32')
		
			# scale between 0-1 to speed up computations
			# print type(pixels_monochrome)
			min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=True)
			pixels_monochrome = min_max_scaler.fit_transform(pixels_monochrome)
			# print len(pixels_monochrome)
			X.append(pixels_monochrome)
			Y.append(lb.transform([im_labels[i]])[0][0])
			# print Y
			# print len(Y)
			#im_aux.show()
			
			if self.verbose:
				#sys.stdout.write("\r Image Type: {0} File Name: {1} Process: {2}/{3}".format(image_type, file_name, count, len(im_paths)))
				#sys.stdout.write("\r File Name: {0} Process: {1}/{2}".format(im_path, count, len(im_paths)))
				sys.stdout.write("\r Process: {0}/{1}".format(i, len(im_paths)))
				sys.stdout.flush()

			i += 1

			# if i == 5:
			# 	break

		train_length = int(round(len(X) * 0.60))
		valid_length = int(round(len(X) * 0.20))
		test_length = int(round(len(X) * 0.20))

		X_train = X[0:train_length]
		X_valid = X[train_length: (train_length + valid_length)]
		X_test = X[-test_length:]

		Y_train = np.array(Y[0:train_length], dtype='float32')
		Y_valid = np.array(Y[train_length:(train_length + valid_length)], dtype='float32')
		Y_test = np.array(Y[-test_length:], dtype='float32')

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


if __name__ == '__main__':
	
	#folder = '/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture-reduced-2/'
	folder = '/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/'
	fe = FetexImage(verbose=True)
	#fe.scale_and_crop_test('/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/n03131574-craddle/n03131574_16.JPEG')
	#print fe.convert_to_bw_and_scale()
	train_set,valid_set,test_set = fe.processImagesPipeline(folder)

