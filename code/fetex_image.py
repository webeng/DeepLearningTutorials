# -*- coding: utf-8 -*-
#from __future__ import division
import math
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
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn import decomposition
from numpy import linalg as LA 
#import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import random
import timeit
import csv
import pandas as pd
import urllib
#from skimage import data, io, filters
import scipy as sp
from scipy.misc import imread
from scipy.signal.signaltools import correlate2d as c2d
import math
from slugify import slugify
from multiprocessing import Process, Pool, Lock, Manager,Queue
import multiprocessing
import copy_reg
import types

def _pickle_method(method):
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	for cls in cls.mro():
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	
	return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class FetexImage(object):
	verbose = None
	width = 128
	height = 128
	model_name = 'images'
	support_per_class = None
	data_path = None
	mode = 'RGB'
	im_index = []
	dataset = None
	df = None
	classes = []
	images = []
	imlist = []
	items_list = {}
	lock = None

	"""docstring for FetexImage"""
	def __init__(self,images=[],verbose = False,support_per_class = None, data_path = None,mode = None,dataset = None, classes = []):
		super(FetexImage, self).__init__()
		self.verbose = verbose
		self.support_per_class = support_per_class
		self.data_path = data_path
		self.mode = mode
		self.dataset = dataset
		self.classes = classes
		self.images = images
		self.lock = Lock()
		self.items_list = []

	def load_images(self,im_paths,imlist,im_index):

		"""Loads all the images paths into a PIL Image object. If an image is in RGBA, then it converts it into a RGB. 
		It return a list of PIL Image Objects"""

		imlist_arr = []
		j = 0
		for im_path in im_paths:
			im = None

			try:
				im = Image.open(im_path)
				#im = imread(im_path)
				#print im.shape
			except Exception, e:
				print e
			
			if im != None:
				try:
					im_aux = np.array(im,dtype=theano.config.floatX)
					im_converted = True
				except TypeError, e:
					im_converted = False
					print e
				
				if im_converted == True:
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
							bn_parsed = os.path.basename(im_path).split("_")
							im_id = int(bn_parsed[0])
							#print im_id
							#Ignore potential duplicates
							#if im_id not in self.im_index:
							if im_id not in im_index:
								im_aux = self.scale_and_crop_img(im)
								# This is for multiprocessing
								im_index.append(im_id)
								imlist.append(np.asarray(im_aux))

								# Uncomment this if you are not using multiprocessing
								# self.im_index.append(im_id)
								# self.imlist.append(np.asarray(im_aux))
								#self.imlist.append(im_aux)
						else:
							print "invalid image: {} size:{}".format(im.filename, im_aux.shape)
		
					except Exception, e:
						#raise e
						print e
	
			# if self.verbose:
			# 	sys.stdout.write("\r Process: {0}/{1}".format(j, len(im_paths)))
			# 	sys.stdout.flush()

			j += 1
	
	def stop_position_per_cpu(self,num_items):
		#Evently distribute the workload of finding similar image amongst all the CPU's
		num_cpus = multiprocessing.cpu_count()

		num_iter = 0
		for i in range(0,num_items):
			num_iter = num_iter + (num_items - i)

		iter_per_cpu = int(math.ceil(num_iter / num_cpus)) + 1

		stop_push = []
		num_iter_i = 0
		for i in range(0,num_items):
			num_iter_i = num_iter_i + (num_items - i)
			if num_iter_i > iter_per_cpu:
				stop_push.append(i)
				num_iter_i = 0

		if stop_push[-1] != num_items:
			stop_push.append(num_items)

		if self.verbose:
			print "num_cpus: {} num_items:{} iter_per_cpu:{}".format(num_cpus,num_items,iter_per_cpu)

		return stop_push

	def load_images_parallel(self,im_paths):
		# Load the image in parallel using as many cpus as possible
		spcpu = self.stop_position_per_cpu(len(im_paths)) # get number of cpu available

		start_time = timeit.default_timer()
		manager = Manager()
		imlist = manager.list()
		im_index = manager.list()

		# Start all the processes with evently distributed load
		p = []
		for i in xrange(0,len(spcpu)):

			stop_i = spcpu[i]
			start_i = 0 if i == 0 else spcpu[i-1] + 1

			print "start_i:{} stop_i:{}".format(start_i,stop_i)

			batch = im_paths[start_i:stop_i]
			p.append(Process(target=self.load_images, args=(batch,imlist,im_index)))
			p[i].start()

		for i in xrange(0,len(spcpu)):
			p[i].join()

		imlist = list(imlist)
		im_index = list(im_index)
		
		if self.verbose:
			end_time = timeit.default_timer()
			elapsed_time = ((end_time - start_time))
			print "Elapsed time {}s loading images".format(elapsed_time)

		return imlist,im_index
	
	def calculate_average_image(self,imlist):
		"""Calculates the average image given PIL Image list"""
		
		N=len(imlist)
		
		if self.mode == 'RGB':
			w,h,c=imlist[0].shape
			arr=np.zeros((h,w,3),theano.config.floatX)
		else:
			w,h=imlist[0].shape		
			arr=np.zeros((h,w),theano.config.floatX)

		for im in imlist:
			imarr=np.array(im,dtype=theano.config.floatX)
			try:
				arr=arr+imarr/N
			except Exception, e:
				print e
			
		arr=np.array(np.round(arr),dtype=np.uint8)
		#arr=np.array(np.round(arr),dtype=theano.config.floatX)
		#average_image=Image.fromarray(arr,mode="RGB")
		average_image=Image.fromarray(arr,mode=self.mode)

		return average_image

	def substract_average_image(self, im, average_image):
		""" Normalise an image by substracting the average image of the dataset. """
		im_minus_avg = np.array(np.round(im),dtype=np.uint8) - np.array(np.round(average_image),dtype=np.uint8)
		#im_minus_avg=Image.fromarray(im_minus_avg,mode="RGB")
		im_minus_avg=Image.fromarray(im_minus_avg,mode=self.mode)
		return im_minus_avg

	def scale_and_crop_img(self,img):
		""" Scale the image to width and height. If the image ratio is not 1:1, then we need the smaller side to be width/height and then crop the center of the image.
		You can uncomment the commented code to transfor the images to Black and White"""

		if img.size[0] < img.size[1]:
			basewidth = self.width
			wpercent = (basewidth/float(img.size[0]))
			hsize = int(float(img.size[1])*float(wpercent))
			img = img.resize((basewidth,hsize), Image.ANTIALIAS)

		else:
			baseheight = self.height
			hpercent = (baseheight/float(img.size[1]))
			wsize = int(float(img.size[0])*float(hpercent))
			img = img.resize((wsize,baseheight), Image.ANTIALIAS)

		if self.mode == 'L':
			img = img.convert('L')

		half_the_width = int(img.size[0] / 2)
		half_the_height = int(img.size[1] / 2)

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
			mypath = self.data_path + self.dataset + '/' + image_type
			onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
			class_support = 0
			for file_name in onlyfiles:
				#print file_name
				if file_name != '.DS_Store':
					im_path = mypath = self.data_path + self.dataset + '/' + image_type + '/' + file_name
					im_paths.append(im_path)
					im_labels.append(image_type)
				class_support += 1
				if self.support_per_class != None and class_support ==  self.support_per_class:
					break

		combined = zip(im_paths, im_labels)
		random.shuffle(combined)
		
		im_paths[:], im_labels[:] = zip(*combined)

		return im_paths,im_labels

	def data_augmentation_and_vectorization(self,imlist, lb,im_labels, average_image = None):
		""" This function applies data augmentation to the images in order to reduce overfitting. 
		Potential transformations: rotation, filtering and normalisation (Norm is not data autmentation). 
		Then it converts the images into numpy arrays and binarizes the labels into numeric classes. 
		Finally we store the arrays and labels into X and Y and return them."""
		X,Y,X_original = [] ,[], []

		i = 0
		for im in imlist:
			im=Image.fromarray(im,mode=self.mode)
			#try:
			#im_ini = im
			im_original = np.asarray(im, dtype=theano.config.floatX) / 256.
			#im = self.substract_average_image(im, average_image)
			#print 'i:{} is a: {}' .format(i,im_labels[i])
			#im.show()
			X_original.append(im_original)

			#Rotations 
			#im_r = im.rotate(15)
			# im_r_2 = im.rotate(-15)
			# im_r_3 = im.rotate(180)
			#im_r.show()
			#im_r_2.show()

			#Filters
			#im_f = im_ini.filter(ImageFilter.DETAIL)
			#im_f = im.filter(ImageFilter.FIND_EDGES)
			
			if self.mode == 'RGB':
				im = np.asarray(im, dtype=theano.config.floatX) / 256.
				#Uncomment this if you want to use cross-correlate for 2D arrays http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.correlate2d.html
				# im = np.asarray(im, dtype=theano.config.floatX)
				# im = sp.inner(im, [299, 587, 114]) / 1000.0
				# im = np.asarray(im, dtype=theano.config.floatX)
				# # normalize per http://en.wikipedia.org/wiki/Cross-correlation
				# im = (im - im.mean()) / im.std()

			if self.mode == 'L':
				# im = np.asarray(im, dtype='float64')
				# im = filters.sobel(im)
				#im = filters.roberts(im)
				im = np.asarray(im, dtype=theano.config.floatX) / 256.
				#im = np.asarray(im, dtype=theano.config.floatX)

			#im = np.asarray(im, dtype=theano.config.floatX)
			
			#im = np.asarray(im, dtype=np.uint8)
			#print im.shape
			#print im.shape
			#im = np.asarray(im, dtype=theano.config.floatX)
			#im = self.flaten_aux(im)
			#print im.shape
			#im = data.coins() # or any NumPy arr
			#print im.shape
			#image = data.coins() # or any NumPy array!
			#print im
			#im = filter.sobel(im)
			#im = filter.roberts(im)

			# im_original = sp.inner(im, [299, 587, 114]) / 1000.0
			# im_original = np.asarray(im_original, dtype=theano.config.floatX)
			# # normalize per http://en.wikipedia.org/wiki/Cross-correlation
			# im = (im_original - im_original.mean()) / im_original.std()
			#print im.shape
			#print edges
			# edges = np.asarray(edges, dtype=np.uint8)
			#Image.fromarray(edges,mode=self.mode).show()

			#print edges

			#im = np.asarray(im, dtype=theano.config.floatX) / 256.

			#print edges.shape
			# io.imshow(im)
			# io.show()
			#im = np.asarray(im, dtype=theano.config.floatX)
			
			# plt.suptitle(im_labels[i], size=16)
			# plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
			# plt.show()
			#im = np.asarray(im, dtype=theano.config.floatX)
			#print im.shape
			#self.reconstructImage(im).show()

			#im_r = np.asarray(im_r, dtype=theano.config.floatX) / 256.
			# im_r_2 = np.asarray(im_r_2, dtype=theano.config.floatX) / 256.
			# im_r_3 = np.asarray(im_r_3, dtype=theano.config.floatX) / 256.
			#im_f = np.asarray(im_f, dtype=theano.config.floatX) / 256.
			
			#im = im.transpose(2, 0, 1)
			#X.append(np.array(im, dtype=theano.config.floatX))
			#X.append(np.array(im_raw, dtype=theano.config.floatX))
			#X.append(im)
			X.append(im)
			# if i % 100 == 0:
			# 	X.append(im)
			#X.append(im_r)
			# X.append(im_r_2)
			# X.append(im_r_3)
			#X.append(im_f)
			#X_original.append(im)

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
			#print lb.transform([im_labels[i]])
			
			label = lb.transform([im_labels[i]])[0][0]
			#print lb.transform([im_labels[i]])
			# label_vector = lb.transform([im_labels[i]])[0]
			# label = np.where( label_vector == 1 )[0][0]
			# print "Label: {}".format(label)
			#print label
			#Y.append(label)
			Y.append(label)
			#Y.append(im_labels[i])	

			
			#Y.append(label)	
			# Y.append(label)	
			# except Exception, e:
			# 	print e
			# 	#raise e

			# if i == 30:
			# 	break

			i += 1
			if self.verbose:
				sys.stdout.write("\r Process: {0}/{1}".format(i, len(imlist)))
				sys.stdout.flush()
		
		# output = open(self.data_path + 'X_original.pkl', 'wb')
		# cPickle.dump(X_original, output,protocol=-1)
		# output.close()

		return X,Y

	def binarize_classes(self):
		lb = preprocessing.LabelBinarizer()
		#classes = ['n07730207-carrot','n04222210-single-bed', 'n00015388-animal','n00017222-flower','n00523513-sport','n01503061-bird','n12992868-fungus']
		#classes = ['n00015388-animal','n00017222-flower','n00523513-sport','n03131574-craddle','n07730207-carrot','n02960352-car', 'n04146614-bus' , 'n09217230-beach' , 'n09238926-cave' ,'n11851578-cactus' , 'n14977504-floor' , 'n04231693-skilift','n03724870-mask']
		#classes = ['n00015388-animal','n09217230-beach','n00017222-flower','n00523513-sport']
		#classes = ['n00015388-animal','n09217230-beach']
		#classes = ["windows-accessories","doors-and-doorways","rooflights-roof-windows","building-areas-systems","ground-substructure","floors-accessories","stairs","ceilings","wall-finishes","roof-structures-finishes","hvac-cooling-systems","insulation","lighting","furniture-fittings","external-works","building-materials","bathroom-sanitary-fittings","structural-frames-walls","drainage-water-supply","communications-transport-security","green-building-products"]
		#classes = ["windows-accessories","doors-and-doorways","rooflights-roof-windows","building-areas-systems","ground-substructure","floors-accessories","stairs","ceilings","wall-finishes","roof-structures-finishes","hvac-cooling-systems","insulation"]
		#classes = ["windows-accessories","doors-and-doorways","rooflights-roof-windows","building-areas-systems","ground-substructure","floors-accessories","stairs","ceilings","wall-finishes","roof-structures-finishes","hvac-cooling-systems","insulation","lighting"]
		#classes = ["windows-accessories","doors-and-doorways","rooflights-roof-windows","building-areas-systems","ground-substructure","floors-accessories"]
		#classes = ["windows-accessories","doors-and-doorways","rooflights-roof-windows"]
		#classes = ["building-areas-systems","ground-substructure","floors-accessories"]
		#class
		#classes = ["stairs","ceilings","wall-finishes","roof-structures-finishes","hvac-cooling-systems","insulation"]
		#classes = ["lighting","furniture-fittings","external-works","building-materials","bathroom-sanitary-fittings"]
		#classes = ["structural-frames-walls","drainage-water-supply","communications-transport-security","green-building-products"]

		
		#classes = ["windows-accessories","doors-and-doorways","rooflights-roof-windows"]
		#classes = ["windows-accessories","doors-and-doorways","lighting"]
		#classes = ["windows-accessories","doors-and-doorways"]
		#classes = ["windows","drainage-water-supply"]
		#classes = ["ceilings","stairs"]
		#classes = ["drainage-water-supply"]
		
		if self.verbose:
			print "Num classes:{}".format(len(self.classes))

		#lb.fit_transform(classes)
		lb.fit(self.classes)

		output = open(self.data_path + 'lb.pkl', 'wb')
		cPickle.dump(lb, output,protocol=-1)
		output.close()

		return self.classes,lb
	
	def flaten_aux(self,V):
		return V.flatten(order='C')
		#return V.flatten(order='F')

	def create_train_validate_test_sets(self,X,Y):
		""" Create Train, validation and test set"""

		print "Size of the original images"

		X = np.asarray(X, dtype=theano.config.floatX)
		
		train_length = int(round(len(X) * 0.60))
		valid_length = int(round(len(X) * 0.20))
		test_length = int(round(len(X) * 0.20))

		X_train = X[0:train_length]
		X_valid = X[train_length: (train_length + valid_length)]
		X_test = X[-test_length:]

		# sample = X_train[0].reshape(64,64)

		# X_train = X_train.transpose(0, 3, 1, 2)
		# X_valid = X_valid.transpose(0, 3, 1, 2)
		# X_test = X_test.transpose(0, 3, 1, 2)

		# X = X.transpose(0, 3, 1, 2)

		X_train = map(self.flaten_aux, X_train)
		X_valid = map(self.flaten_aux, X_valid)
		X_test = map(self.flaten_aux, X_test)

		# X = map(self.flaten_aux, X)

		#print X_train.shape
		#X = X.transpose(0, 3, 1, 2)
		# X = np.asarray(X, dtype=theano.config.floatX)
		# X = X.reshape((21, 3, 64, 64))
		# print X.shape
		# #X_train = X_train.transpose(0, 3, 1, 2)
		# #print X[0].
		# im = Image.fromarray(X[0],mode="RGB")
		# im.show()
		#self.reconstructImage(X[0]).show()
		# sample = X_train[0].reshape(64,64)
		# Image.fromarray(sample,mode="L").show()

		#X = map(self.flaten_aux, X)

		# X_train = X[0:train_length]
		# X_valid = X[train_length: (train_length + valid_length)]
		# X_test = X[-test_length:]

		Y_train = Y[0:train_length]
		Y_valid = Y[train_length:(train_length + valid_length)]
		Y_test = Y[-test_length:]

		#pkl_file = open( '../data/lb.pkl', 'rb')
		#lb = cPickle.load(pkl_file)

		#arr = np.array(np.round((X_train[0] * 256).reshape((64,64))),dtype=np.uint8)
		# Image.fromarray(arr,mode="L").show()
		# print lb.classes_
		# print Y_train[0]

		train_set = [X_train,Y_train]
		valid_set = [X_valid,Y_valid]
		test_set = [X_test,Y_test]
		input = [X,Y]

		if self.verbose:
			print "X_train {} X_validation {} X_test {}".format(len(X_train),len(X_valid),len(X_test))
			print "Y_train {} Y_validation {} Y_test {}".format(len(Y_train),len(Y_valid),len(Y_test))

		output = open(self.data_path + 'train_set.pkl', 'wb')
		cPickle.dump(train_set, output,protocol=-1)
		output.close()

		output = open(self.data_path + 'valid_set.pkl', 'wb')
		cPickle.dump(valid_set, output,protocol=-1)
		output.close()

		output = open(self.data_path + 'test_set.pkl', 'wb')
		cPickle.dump(test_set, output,protocol=-1)
		output.close()
		
		return train_set,valid_set,test_set

	def reconstructImage(self,arr):
		
		""" Reconstruct an image from array """
		arr = arr * 256
		arr = np.array(np.round(arr),dtype=np.uint8)
		#arr = np.array(arr,dtype=np.uint8)

		# We need to transpose the array because we flatten X by columns
		#arr = arr.T
		#a = arr.reshape((self.width, self.height,3))
		
		if self.mode == 'L':
			a = arr.reshape((self.width, self.height))
		else:
			a = arr.reshape((self.width, self.height,3))

		#a = arr.reshape((3,self.width, self.height))		
		#a = arr.transpose(0, 3, 1, 2)

		im = Image.fromarray(a,mode=self.mode)

		return im

	def ImagePipeline(self,cnn_pipe = False, batch_index = None):
		""" This is the main pipeline to process the images"""
		
		if self.verbose:
			print "...createFolderStructure"

		self.createFolderStructure()

		if self.verbose:
			print "...downloadImages"

		self.downloadImages()

		if self.verbose:
			print "...binarize_classes"

		classes, lb = self.binarize_classes()

		if self.verbose:
			print "...load_paths_and_labels"

		im_paths, im_labels = self.load_paths_and_labels(classes)

		if self.verbose:
			print "...load_images"
				
		# Uncomment this if you just want to use one cpu		
		#imlist = self.load_images(im_paths,cnn_pipe)
		#self.load_images(im_paths,[],[])
		#imlist = self.imlist

		imlist, self.im_index = self.load_images_parallel(im_paths)
		#print len(imlist)

		# Sort the list by index so we don't have to do as many iteration in finding similar
		#if not cnn_pipe:
		zipped = zip(self.im_index, imlist)
		zipped_sorted = sorted(zipped, key=lambda x: x[0])
		self.im_index , imlist = zip(*zipped_sorted)

		average_image = None
		if cnn_pipe:
			if self.verbose:
				print "...calculate_average_image"
			average_image = self.calculate_average_image(imlist)

		if self.verbose:
			print "\n...data_augmentation_and_vectorization\n"
		
		#print imlist

		X,Y = self.data_augmentation_and_vectorization(imlist,lb,im_labels,average_image)

		output = open( self.data_path + 'im_index.pkl', 'wb')
		cPickle.dump(self.im_index, output,protocol=-1)
		output.close()

		if self.verbose:
			print "...dimReductionSdA"
		
		X = self.dimReductionSdA(X)
		# print X[0][0:3]
		# print X[1][0:3]
		#X = self.dimReduction(X)

		output = open( self.data_path + 'X_compressed_'+str(batch_index)+'.pkl', 'wb')
		cPickle.dump(X, output,protocol=-1)
		output.close()

		output = open( self.data_path + 'im_index_' + str(batch_index) + '.pkl', 'wb')
		cPickle.dump(self.im_index, output,protocol=-1)
		output.close()

		if cnn_pipe:
			if self.verbose:
				print "\n...create_train_validate_test_sets\n"
			train_set,valid_set,test_set =  self.create_train_validate_test_sets(X, Y)
			return train_set,valid_set,test_set
		else:

			if self.verbose:
				print "\n...similarImages\n"

			df, duplicated_images = self.similarImages(X)
		
			return df,duplicated_images

	def removeDuplicates(self,seq):
		seen = set()
		seen_add = seen.add
		return [ x for x in seq if not (x in seen or seen_add(x))]
	
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

		with open(self.data_path + 'categories.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			next(reader, None)  # skip the headers
			for row in reader:
				directory = self.data_path + 'categories/' + str(row[1])
				if not os.path.exists(directory):
					os.makedirs(directory)
	
	def downloadImages(self):

		""" query to get all the images and slug
				SELECT documents.`id`, documents.`name`, documents.`url`, categories.`slug`
		FROM documents 
		JOIN products ON documents.`product_id` = products.id
		JOIN `categories` ON categories.id = products.`category_id`
		WHERE documents.type = 'photo'
		AND products.`category_id` IS NOT NULL
		"""
		i = 0
		for im in self.images:
			# Let's get the file extension and file name and make the final file path. 
			# We need to do this to slugify the file name and avoid errors when loading images
			file_name, file_extension = os.path.splitext(im['url'])
			file_name = file_name.split("/")[-1]

			file_path = self.data_path + self.dataset + "/" + im['slug'] + '/' + str(im['id']) + '_' + slugify(file_name) + file_extension

			# If file is not in the file path, then download from the url
			if not os.path.exists(file_path):
				try:
					urllib.urlretrieve(im['url'], file_path )
					print "i:{} url:{}".format(i,im['url'])
				except Exception, e:
					print e
			i += 1
	
	def readImagesCSV(self):
		images = []
		with open(self.data_path + '/images.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			next(reader, None)  # skip the headers
			i = 0
			for row in reader:

				filename, file_extension = os.path.splitext(row[2])
				file_name = filename.split("/")[-1]

				file_path = self.data_path + self.dataset + "/" + row[3] + '/' + row[0] + '_' + slugify(file_name) + file_extension
				#print row[3]
				aux = {}
				aux['url'] = row[2]
				aux['file_path'] = file_path
				images.append(aux)

		return images

	def downloadImagesCSV(self,images):

		i = 0
		for im in images:
			# If file is not in the file path, then download from the url
			if not os.path.exists(im['file_path']):
				try:
					#print "i:{} url:{}".format(i,im['url'])
					urllib.urlretrieve(im['url'], im['file_path'] )
				except Exception, e:
					print e
			i += 1

	def cosine_distance(self,a, b):
		dot_product =  np.dot(a,b.T)
		cosine_distance = dot_product / (LA.norm(a) * LA.norm(b))
		return cosine_distance

	def load_SdA_weights(self,num_SdA_layer):
		SdA_layers_W = []
		SdA_layers_b = []

		for i in range(0,num_SdA_layer):
			pkl_file = open(self.data_path + 'dA_layer'+str(i)+'_W.pkl', 'rb')
			W_aux = cPickle.load(pkl_file)
			W_aux = np.asarray(W_aux, dtype=theano.config.floatX)
			SdA_layers_W.append(W_aux)
			
			pkl_file = open(self.data_path + 'dA_layer'+str(i)+'_b.pkl', 'rb')
			b_aux = cPickle.load(pkl_file)
			b_aux = np.asarray(b_aux, dtype=theano.config.floatX)
			SdA_layers_b.append(b_aux)

		return SdA_layers_W, SdA_layers_b

	def dimReductionSdA(self,X):
		import theano
		import theano.tensor as T
		from scipy.special import expit

		if self.verbose:
			print "... flattening "

		X = map(self.flaten_aux, X)		
		X = np.asarray(X, dtype=theano.config.floatX)

		# Get activations unit for layer 0
		#expit = sigmoid
		SdA_layers_W, SdA_layers_b = self.load_SdA_weights(2)
		W_0 = SdA_layers_W[0]
		b_0 =  SdA_layers_b[0]

		print "X shape: {} W_0 shape:{} b_0 shape:{}".format(X.shape,W_0.shape,b_0.shape)

		da1output = expit(np.dot(X,W_0) + b_0)

		W_1 = SdA_layers_W[1]
		b_1 =  SdA_layers_b[1]

		print "da1output shape: {} W_0 shape:{} b_0 shape:{}".format(da1output.shape,W_1.shape,b_1.shape)

		da2output = expit(np.dot(da1output,W_1) + b_1)
		#print da2output[0][0:10]
		return da2output
		
		"""
		from SdA_v2 import SdA
		#uncomment this to check duplicates
		# seen = set()
		# uniq = []
		# for x in self.im_index:
		# 	if x not in seen:
		# 		uniq.append(x)
		# 		seen.add(x)
		# 	else:
		# 		print "Repeated image_id: {}".format(x)
		# print self.im_index
		# print uniq
		numpy_rng = np.random.RandomState(89677)

		sda2 = SdA(
			numpy_rng=numpy_rng,
			n_ins=128 * 128 * 3,
			hidden_layers_sizes=[1000, 1000],
			n_outs=21,
			data_path=self.data_path
		)

		#print X.get_value(borrow = True).shape
		#sda2.load_weights()

		W_0 = sda2.dA_layers[0].W.get_value(borrow=True)
		b_0 =  sda2.dA_layers[0].b.get_value(borrow=True)

		W_0 = np.asarray(W_0, dtype=theano.config.floatX)
		b_0 = np.asarray(b_0, dtype=theano.config.floatX)
		
		print "X shape: {} W_0 shape:{} b_0 shape:{}".format(X.shape,W_0.shape,b_0.shape)

		#print expit(np.dot(X,W_0) + b_0 )
		da1output = expit(np.dot(X,W_0) + b_0)
		#print da1output

		W_1 = sda2.dA_layers[1].W.get_value(borrow=True)
		b_1 =  sda2.dA_layers[1].b.get_value(borrow=True)

		W_1 = np.asarray(W_1, dtype=theano.config.floatX)
		b_1 = np.asarray(b_1, dtype=theano.config.floatX)

		print "da1output shape: {} W_0 shape:{} b_0 shape:{}".format(da1output.shape,W_1.shape,b_1.shape)
		# print da1output.shape
		# print W_1.shape
		# print b_1.shape
		#print expit(np.dot(X,W_0) + b_0 )
		da2output = expit(np.dot(da1output,W_1) + b_1)
		#print expit(np.dot(X,W_0 + b_0))
		return da2output
		#print da2output[0][0:10]"""

		#print da2output
		# X = da2output
		# print X
		# return X
		#return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
		"""
		X = theano.shared(np.asarray(X,
			dtype=theano.config.floatX),
			borrow=True)

		x = T.matrix('x')
		index_1 = T.lscalar()    # index to a [mini]batch
		index_2 = T.lscalar()    # index to a [mini]batch
		getHV = sda2.dA_layers[0].get_hidden_values(x)
		getHiddenValues = theano.function(
			[index_1,index_2],
			getHV,
			givens={
				x: X[index_1:index_2]
			}
		)
		#print X.get_value()[0].shape
		#print getHiddenValues(0,len(X.get_value(borrow=True)))
		
		da1output = T.matrix('da1output')
		getHV2 = sda2.dA_layers[1].get_hidden_values(da1output)
		getHiddenValues2 = theano.function(
			[da1output],
			getHV2
		)

		#print getHiddenValues2(getHiddenValues(0,1)).shape
		X = getHiddenValues2(getHiddenValues(0,len(X.get_value(borrow=True))))
		print X[0][0:10]
		return X"""

	def dimReduction(self,X):

		seen = set()
		uniq = []
		for x in self.im_index:
			if x not in seen:
				uniq.append(x)
				seen.add(x)
			else:
				print "Repeated image_id: {}".format(x)
		print len(self.im_index)
		print len(uniq)

		if self.verbose:
			print "... flattening "
		X = map(self.flaten_aux, X)
		
		# print len(self.im_index)
		# print len(set(self.im_index))

		#X = X.transpose(0, 3, 1, 2)
		# print X[0].shape
		
		# TODO: Do PCA in batches as we don't have lots of memory. Batch_size = 5000
		if self.verbose:
			print "... training PCA"
		# batch_size = 2000
		# num_batches = int(len(X) / batch_size) + 1
		# if self.verbose:
		# 	print "... training PCA in batches of size:{} num_batches: {}".format(batch_size,num_batches)

		# X_aux = []
		# #X_aux = np.asarray(X_aux)
		# for i in xrange(0,num_batches):
		# 	X_batch = X[ i * batch_size : (i+1) * batch_size]
			
		# 	start_time = timeit.default_timer()
		# 	#pca = PCA(n_components=0.95)
		# 	pca = RandomizedPCA(n_components=676)
		# 	pca.fit(X_batch)
			
		# 	end_time = timeit.default_timer()
		# 	elapsed_time = ((end_time - start_time) / 60.)
		# 	print "Elapsed time pca {}m".format(elapsed_time)

		# 	print "Total variance kept = {}".format(sum(pca.explained_variance_ratio_))
		# 	print "... pca.transform"

		# 	#X = pca.transform(X)
		# 	if len(X_aux) == 0:
		# 		X_aux = pca.transform(X_batch)
		# 	else:
		# 		X_aux = np.concatenate((X_aux, pca.transform(X_batch)),axis=0)
		# 	# X_aux.append(pca.transform(X_batch))
		# 	# print X_aux[0].shape
		
		# X = X_aux
		# # X = np.asarray(X)
		start_time = timeit.default_timer()
		
		#pca = RandomizedPCA(n_components=676)
		pca = RandomizedPCA(n_components=676)
		pca.fit(X)
		X = pca.transform(X)
		
		end_time = timeit.default_timer()
		elapsed_time = ((end_time - start_time) / 60.)
		print X.shape
		print X[0].shape

		print "... pca fit_transform time:{}m".format(elapsed_time)
		print "Total variance kept = {}".format(sum(pca.explained_variance_ratio_))

		# print X[0][0]
		# output = open( self.data_path + 'pca_' + self.mode + '.pkl', 'wb')
		# cPickle.dump(pca, output,protocol=-1)
		# output.close()

		# pkl_file = open( self.data_path + 'pca.pkl', 'rb')
		# pca = cPickle.load(pkl_file)
		
		#Uncomment this if your want to work with 2D Correlation
		#print pca.explained_variance_ratio_

		# side_length = int(math.sqrt(float(X[0].shape[0])))
		# print "side lenght: {}".format(side_length)

		# X_aux = []
		# for x in X:
		# 	x = x.reshape(side_length,side_length)
		# 	X_aux.append(x)

		# X = X_aux
		#X = X.reshape(side_length,side_length)



		return X
	
	def appendToCSV(self,items_list,addHeader = True, file_name = '_data.csv',duplicates = False):
		if self.verbose:
			print "...appendToCSV file_name:{}".format(file_name)
		
		with open(self.data_path + self.model_name + file_name, 'wb') as csvfile:
			
			writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			#writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			#reader = csv.reader(csvfile_read, delimiter=',', quotechar='"')
			if addHeader:
				writer.writerow(['item_id_x' , 'item_id_y' , 'distance'])
			
			for item in items_list:
				if not duplicates:
					writer.writerow([item['item_id_x'], item['item_id_y'], item['distance']])
				else:
					writer.writerow([item])
	
	def concatenateCSVFiles(self,num_cpus):
		csvfile_write = open(self.data_path + self.model_name + '_data.csv', 'wb')
		#writer = csv.writer(csvfile_write, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer = csv.writer(csvfile_write, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

		for i in xrange(0,num_cpus):
			file_path = self.data_path + self.model_name +  '_data_'+str(i)+'.csv'
			if i == 0:
				writer.writerow(['item_id_x' , 'item_id_y' , 'distance'])

			with open(file_path, 'rb') as csvfile_read:
				reader = csv.reader(csvfile_read, delimiter=',', quotechar='"')
			
				for row in reader:
					writer.writerow([row[0], row[1], row[2]])

			#Remove file
			os.remove(file_path)

	def concatenateXandIndex(self,classes):
		X_compressed = []
		im_index_all = []

		for i in xrange(0,len(classes)):
			x_path = self.data_path + 'X_compressed_' + str(i) + '.pkl'
			index_path = self.data_path + 'im_index_' + str(i) + '.pkl'
			if os.path.exists(x_path):

				pkl_file = open(x_path, 'rb')
				X = cPickle.load(pkl_file)
	
				pkl_file = open(index_path, 'rb')
				im_index = cPickle.load(pkl_file)
				im_index = list(im_index)
	
				for i in range(0,len(X)):
					X_compressed.append(X[i])
					im_index_all.append(im_index[i])
				
				os.remove(x_path)
				os.remove(index_path)

		output = open(self.data_path + 'X_compressed.pkl', 'wb')
		cPickle.dump(X_compressed, output,protocol=-1)
		output.close()

		output = open(self.data_path + 'im_index_all.pkl', 'wb')
		cPickle.dump(im_index_all, output,protocol=-1)
		output.close()

	def concatenateDuplicatedItems(self,num_cpus):
		duplicated_items = []
		for i in xrange(0,num_cpus):
			
			file_path = self.data_path + self.model_name +  '_duplicated_items_'+str(i)+'.csv'
			
			if os.path.exists(file_path):
				with open(file_path, 'rb') as csvfile_read:
					reader = csv.reader(csvfile_read, delimiter=',', quotechar='"')
					for row in reader:
						duplicated_items.append(int(row[0]))
			
				#Remove file
				os.remove(file_path)

		return duplicated_items

	def similarImages2(self,X,start_i,stop_i,cpu_index):

		"""
		These features are binarized (fc6) and sparsified (fc8) 
		for representation efficiency and compared using Hamming distance (fc6) and
		cosine similarity (fc8), respectively.

		"""

		duplicated_items = []
		items_list = []
		# print "this is going to iterate for:{}".format(len(xrange(start_i,stop_i)))
		for i in xrange(start_i,stop_i):
			a = X[i]

			#print a
			# We just need to fill the upper diagonal
			# Remember to uncomment this if you are not using the autoencoder
			#for j in xrange(i,len(X)):
			#print "this is going to iterate for:{}".format(len(xrange(i,len(X))))
			for j in xrange(i,len(X)):
				aux = {'item_id_x' : None , 'item_id_y' : None , 'distance' : None}
				b = X[j]
				#uncomment this if you want to use the cosine distance
				d = self.cosine_distance(a,b)
				#Uncomment this if you want to use cross-correlate for 2D arrays http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.correlate2d.html
				# c = c2d(a, b, mode='same')
				# d = c.max()
				#Uncomment this for euclidean distance
				# d = dist.euclidean(a, b)
				#d = dist.cityblock(a, b)

				# Uncomment this if you want to use the euclidean distance
				# if d == 0 and i != j:
				# 	duplicated_items.append(self.im_index[j])

				# if i == j or d == 0:
				# 	d = -np.inf

				# Uncomment this if you are going to use the cosine distance
				if d == 1 and i != j:
					#print "Im1 {} Im2 {}".format(self.im_index[i], self.im_index[j])
					duplicated_items.append(self.im_index[j])

				if i == j or d == 1:
					d = -np.inf

				aux['item_id_x'] = self.im_index[i]
				aux['item_id_y'] = self.im_index[j]
				aux['distance'] = d
				items_list.append(aux)
		#print len(duplicated_items)
		self.appendToCSV(items_list,addHeader=False,file_name='_data_'+str(cpu_index)+'.csv',duplicates=False)
		
		print "cpu_index:{} num_iterations:{}".format(cpu_index,len(items_list))

		if len(duplicated_items) > 0:
			self.appendToCSV(duplicated_items,addHeader=False,file_name='_duplicated_items_'+str(cpu_index)+'.csv',duplicates=True)
			
	def similarImages(self,X):

		# X = self.dimReduction(X)
		# elif SdA:
		# 	X = self.dimReductionSdA(X)

		# if self.verbose:
		# 	print "... Scaling to the positive space"

		# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=True)
		# X = min_max_scaler.fit_transform(X)
		
		if self.verbose:
			print "... finding similar images"

		start_time = timeit.default_timer()
		spcpu = self.stop_position_per_cpu(len(X))

		p = []
		for i in xrange(0,len(spcpu)):
			
			stop_i = spcpu[i]
			start_i = 0 if i == 0 else spcpu[i-1] + 1

			print "start_i:{} stop_i:{}".format(start_i,stop_i)
			p.append(Process(target=self.similarImages2, args=(X,start_i,stop_i,i)))
			p[i].start()

		for i in xrange(0,len(spcpu)):
			p[i].join()

		#Concatenate csv files
		self.concatenateCSVFiles(len(spcpu))
		duplicated_items = self.concatenateDuplicatedItems(len(spcpu))

		if self.verbose:
			#print "len items_list:{}".format(len(items_list))
			end_time = timeit.default_timer()
			elapsed_time = ((end_time - start_time))
			print "Elapsed time {}s finding similar images".format(elapsed_time)

		if self.verbose:
			print "There are {} duplicates.".format(len(duplicated_items))

		duplicated_items = self.removeDuplicates(duplicated_items)

		if self.verbose:
			print "...loading .csv to pandas dataframe"
		
		df = pd.read_csv(self.data_path + self.model_name + '_data.csv' ,sep = ' ')
		
		#df[['user_id']] = df[['user_id']].astype(str)
		#df[['user_id']] = df[['user_id']].astype(str)
		df = df.pivot(index="item_id_x",columns="item_id_y",values="distance")
		df = df.fillna(0)
		df = df.sort_index(axis = 0, ascending=[True])
		df = df.sort_index(axis = 1, ascending=[True])
		self.df = df

		print df.head()

		#item_id = 47613
		#item_id = 2218
		# item_id = 47574
		# i = self.im_index.index(item_id)
		# # recommendations = df[df[item_id] > 0].sort_values(by=[item_id],ascending = False)[item_id][0:8].index.values.tolist()
		# # print recommendations

		# index_x = df.iloc[0:i,i].index
		# index_y = df.iloc[i,i:df.index.values.size].index

		# values_x = df.iloc[0:i,i].values
		# values_y = df.iloc[i,i:df.index.values.size].values

		# index = np.concatenate((index_x, index_y),axis=0)
		# values = np.concatenate((values_x,values_y),axis=0)

		# zipped = zip(index,values)
		# zipped_sorted = sorted(zipped, key=lambda x: x[1])[::-1][0:8]
		# index , values = zip(*zipped_sorted)
		# recommendations = list(index)
		# print recommendations

		if self.verbose:
			print "...storing to HDFS path : {}".format(self.data_path)
		
		#store = pd.HDFStore(self.storage_path + 'df_' + self.model_name + '.h5', 'w')
		store = pd.HDFStore(self.data_path + 'df_' + self.model_name + '.h5', 'w')
		store['df'] = df
		store.close()

		return df,duplicated_items

	def combine_df(self,df1,df2,weight_1,weight_2):
		self.df = (df1 * weight_1) + (df2 * weight_2)
		return self.df 

	def showRecommendations(self):
		#rn_im_index  = np.where( df_index == 10561)[0][0] #similar color but no similar shape
		"""
		Output shape
		[(27329, 0.10757738173046448), (27103, 0.099139333741020991), (27088, 0.095615475573191125), (27323, 0.094846251648387933), (30034, 0.094423612678069882), (29937, 0.094423612678069882), (52539, 0.093585170222998962), (27402, 0.092435271902316984)]

		Output Color
		[(10578, 0.70587217797904644), (10594, 0.58090272800456222), (94048, 0.48265102426805062), (61846, 0.47927602173685019), (61760, 0.47009989997940232), (57461, 0.46876431655701217), (61720, 0.46771377897219141), (61715, 0.46166750232534359)]

		"""
		
		#rn_im_index  = np.where( df_index == 22472)[0][0] # similar color but no similar shape
		"""
		Output shape
		[(61706, 0.16241728944546732), (94073, 0.15613203034271395), (61836, 0.15494992784841455), (61835, 0.15494992784841452), (61825, 0.15163383319000062), (61745, 0.15031672266647675), (26848, 0.14479933826475058), (61760, 0.14353241349060006)]

		Output Color
		[(22492, 0.72863097869032856), (22482, 0.66834821692729429), (3351, 0.45135804324105538), (29982, 0.40733726762782918), (85603, 0.40595375826379132), (22502, 0.38204339162468243), (29913, 0.36735985661014864), (29581, 0.3669268043422747)]

		"""
		
		#rn_im_index  = np.where( df_index == 26746)[0][0] #Similar shape and similar color

		"""
		Output shape
		[(27380, 0.1817530749164192), (29457, 0.1353165149065198), (1336, 0.12885937891206711), (27355, 0.12241573468787358), (29704, 0.12009259771972887), (29603, 0.11196184515165516), (29594, 0.11196184515165516), (26809, 0.11097441686854403)]

		Output Color
		[(26809, 0.80634030626051745), (27380, 0.79789790693763663), (27355, 0.79542468562323521), (27018, 0.74331190002098657), (27197, 0.73454915804315535), (26913, 0.73410853271216192), (26905, 0.73410853271216192), (27617, 0.73098284820738935)]

		"""

		#rn_im_index = np.where( df_index == 27288)[0][0] #blurry image
		#rn_im_index = np.where( df_index == 27294)[0][0] # Similar Color and similar shape
		"""
		Output shape
		[(27133, 0.35485652442453264), (27128, 0.32115384345167203), (27151, 0.25627343126278629), (27145, 0.25366123246450772), (27237, 0.25131923154633229), (27303, 0.22385072157466906), (27139, 0.22229444866797674), (27299, 0.22049959456469045)]

		Output Color
		[(27133, 0.96240728970715483), (27128, 0.96009243888171958), (27145, 0.94268324228267275), (27303, 0.93286490646887354), (27139, 0.9244608465512546), (27237, 0.87199166625029467), (27049, 0.86531150055386774), (27066, 0.86139090244063599)]

		"""

		#rn_im_index = np.where( df_index == 52528)[0][0] # some have similar shape and some have similar color
		"""
		Output shape
		[(93975, 0.31989999912901967), (61835, 0.31528273207820834), (61836, 0.31528273207820828), (61745, 0.31261425625988493), (61825, 0.31226105280375738), (61706, 0.31006537435901937), (61760, 0.29497111365575518), (94073, 0.28643748527418661)]
		
		Output Color
		[(52542, 0.7633360888150692), (27402, 0.7582411610565466), (59301, 0.71242045321505865), (27329, 0.69968585913071302), (52539, 0.6996578131078881), (27335, 0.69215065941368603), (52469, 0.69152133535379212), (52473, 0.68799897765402473)]

		Output c2d
		[(85620, 39705.292103093299), (52469, 38947.56038916672), (93975, 37706.480789897578), (52542, 37604.001320837888), (27402, 36709.321927197598), (27118, 36164.067396937884), (63718, 35906.648243400079), (63709, 35906.648243400079)]
	

		"""
		# Similar in color but dissimilar in shape
		#rn_im_index = np.where( df_index == 94380)[0][0] # Similar with color. Similar with shape. Very good with shape. Good Recommendations 52469(Shape) 94383 (color)
		
		"""
		Output shape
		[(52469, 0.22380221768394279), (61836, 0.17343131445222859), (61835, 0.17343131445222859), (61825, 0.1713416617900273), (61745, 0.16700001977657994), (35922, 0.16614680579871874), (61715, 0.16380442450621885), (61706, 0.16194776280945139)]
		
		Output Color
		[(94383, 0.69238692936637536), (26960, 0.58939898313472816), (26957, 0.58939898313472816), (29412, 0.58436143235370375), (29371, 0.58436143235370375), (29453, 0.5745231714319865), (29616, 0.57270906625007156), (29970, 0.57018718322031081)]

		Output c2d
		[(94383, 37226.57203206882), (52558, 37007.251051234598), (26960, 36448.333956681076), (26957, 36448.333956681076), (1441, 36380.413117473567), (50197, 35994.006084886816), (94057, 35671.971168930344), (27533, 35061.385308567049)]
	
		"""

		#rn_im_index = np.where( df_index == 94080)[0][0] # some have similar shape and some have similar color
		"""
		Output c2d
		[(57755, 29305.613736454678), (61797, 28828.064153886309), (61731, 28828.064153886309), (29417, 27874.375538422293), (63771, 27596.578857622582), (63765, 27596.578857622582), (63758, 27442.936837903482), (63750, 27442.936837903482)]

		"""

		# Completely random image that doesn't have similar images
		#rn_im_index = np.where( df_index == 1334)[0][0]
		df = self.df
		df_index = df.index.values
		rn_im_index  = random.randint(0, df.shape[0])

		print "random image index: {} id:{}".format(rn_im_index, df_index[rn_im_index])

		i = rn_im_index
		index_x = df.iloc[0:i,i].index
		index_y = df.iloc[i,i:df.index.values.size].index

		values_x = df.iloc[0:i,i].values
		values_y = df.iloc[i,i:df.index.values.size].values

		index = np.concatenate((index_x, index_y),axis=0)
		values = np.concatenate((values_x,values_y),axis=0)

		zipped = zip(index,values)
		zipped_sorted = sorted(zipped, key=lambda x: x[1])[::-1][0:8]
		#zipped_sorted = sorted(zipped, key=lambda x: x[1])[0:8]
		print zipped_sorted
		index , values = zip(*zipped_sorted)
		#print index
		top_n_similar_images = map(int,list(index))
		#return df, duplicated_items

		# Filter out threshold less than 0.5
		#if self.mode == 'RGB':
		index_aux = []
		i = 0
		for im_id in top_n_similar_images:
			if self.mode == 'RGB' and values[i] > 0.5:
				index_aux.append(im_id)
			elif self.mode == 'L' and values[i] > 0.1:
				index_aux.append(im_id)
			i += 1

		top_n_similar_images = index_aux

		if len(top_n_similar_images) > 0 or self.mode == 'L':
		
			#print top_n_similar_images
			top_n_similar_images = self.removeDuplicates(top_n_similar_images)
			#print top_n_similar_images
	
			#top_n_similar_images = df.sort_values(by=[rn_im_index],ascending = False).loc[:,rn_im_index][0:10].index.values
			
			output = open(self.data_path + 'X_original.pkl', 'r')
			X_original = cPickle.load(output)
			output.close()
			
			#print top_n_similar_images[0]
			index = np.asarray(index,dtype='int64')
			
			if self.mode == 'RGB':
				self.reconstructImage(X_original[rn_im_index]).show()
			elif self.mode == 'L':
				im_base = X_original[rn_im_index] * 256
				im_base = np.asarray(im_base, dtype='float64')
				im_base = filter.sobel(im_base)
	
				io.imshow(im_base)
				io.show()	

			for i in xrange(0,len(top_n_similar_images)):
				index_i = np.where( df_index == top_n_similar_images[i])[0][0]

				if self.mode == 'L':
					im_i = X_original[index_i] * 256
					im_i = np.asarray(im_i, dtype='float64')
					im_i = filter.sobel(im_i)
	
					io.imshow(im_i)
					io.show()

				elif self.mode == 'RGB':
					self.reconstructImage(X_original[index_i]).show()
		else:
			print "There are no image higher than the minimum threshold"

	#def getSimilarItems(self,item_id = None,user_id = None, num_recommendations = 10, model_name = 'products'):
	def getRecommendations(self,image_url, num_recommendations = 10, model_name = 'images'):
		# This method gets n similar images given a image url. It downloads the image that we staged in our S3.

		# Preproces the image by applyning a reduced version of the method load_image.
		file_path = self.data_path + 'file.jpg'
		urllib.urlretrieve(image_url, file_path )
		im = Image.open(file_path)

		try:
			im_aux = np.array(im,dtype=theano.config.floatX)
			im_converted = True
		except TypeError, e:
			im_converted = False
			print e
				
		if im_converted == True:
			try:
				if im_aux.shape[2] == 4:
					background = Image.new("RGB", im.size, (255, 255, 255))
					background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
					im = background
			except Exception, e:
				print e

		im = self.scale_and_crop_img(im)
		im = np.asarray(im, dtype=theano.config.floatX) / 256.

		# Reduce the dimension of the dataset using a denoise Autoencoder that we previously trained
		a = self.dimReductionSdA([im])[0]

		# Load the compressed representation of our images
		pkl_file = open( self.data_path + 'X_compressed.pkl', 'rb')
		X_compressed = cPickle.load(pkl_file)

		# Load the document index of the previous loaded images. The id is the document id that we have in the database
		pkl_file = open( self.data_path + 'im_index_all.pkl', 'rb')
		im_index = cPickle.load(pkl_file)
		
		X = np.asarray(X_compressed,dtype=theano.config.floatX)

		# Compare the given image with all the images in the dataset and store the distances in an auxiliar list
		distances = np.zeros(len(X))
		for i in xrange(0,len(X)):
			b = X[i]
			d = self.cosine_distance(a,b)
			distances[i] = d

		# Sort the indices and get the indices of the sorted array so we can get the document id later
		distances_index = np.argsort(distances)[::-1]

		# Top 3 matches
		# print im_index[distances_index[0]]
		# print im_index[distances_index[1]]
		# print im_index[distances_index[2]]

		im_index_aux = []
		for i in xrange(0,len(distances_index)):
			im_index_aux.append(im_index[distances_index[i]])
			if i == num_recommendations:
				break

		im_index = im_index_aux

		return im_index, model_name

	""" get products per category

	SELECT COUNT(*) as total, products.type, categories.`name`
		FROM documents 
		JOIN products ON documents.`product_id` = products.id
		JOIN `categories` ON categories.id = products.`sub_category_id`
		WHERE documents.type = 'photo'
		AND products.`type` IS NOT NULL
		GROUP BY products.`sub_category_id`
		ORDER BY total DESC

	"""
if __name__ == '__main__':
	import cProfile
	import re
	#data_path = '/Applications/MAMP/htdocs/DeepLearningTutorials/data/'
	data_path = '/Applications/MAMP/htdocs/AI/ml_server/data/images/localhost/'
	#data_path = '/Applications/MAMP/htdocs/DeepLearningTutorials/data/'
	#folder = '/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/'
	classes = ["stairs","ceilings","building-areas-systems"]
	# classes = ["floors-accessories","ground-substructure", "building-areas-systems",
 #            "furniture-fittings","stairs","rooflights-roof-windows",
 #            "wall-finishes","ceilings","communications-transport-security",
 #            "external-works","green-building-products","insulation",
 #            "building-materials","hvac-cooling-systems","drainage-water-supply",
 #            "roof-structures-finishes","lighting","structural-frames-walls",
 #            "doors-and-doorways","bathroom-sanitary-fittings","windows-accessories"]
	#fe = FetexImage(verbose=True,support_per_class=1000,data_path=data_path, dataset='categories' ,mode='L', classes = classes)
	fe = FetexImage(verbose=True,support_per_class=800,data_path=data_path, dataset='categories' ,mode='RGB', classes = classes)
	#fe.scale_and_crop_test('/Applications/MAMP/htdocs/DeepLearningTutorials/data/cnn-furniture/n03131574-craddle/n03131574_16.JPEG')
	#print fe.convert_to_bw_and_scale()
	#train_set,valid_set,test_set = fe.processImagesPipeline()
	#fe.ImagePipeline(cnn_pipe=False)
	#fe.getRecommendations('http://www.wood-floor.co.uk/images/doors/door-headers/Wood_Door-Liner_skirting.jpg',10)
	# fe.getRecommendations('https://s3-eu-west-1.amazonaws.com/specifiedbypro/4609/8889/calibre-windows_UPVC-Doors-and-Panels_Images_Image032.jpg',10)
	#fe.getRecommendations('http://hamptonroadshappyhour.com/sites/default/files/aa_carrotini_3.jpg',10)
	cProfile.run("fe.getRecommendations('https://s3-eu-west-1.amazonaws.com/specifiedbypro/4609/8889/calibre-windows_UPVC-Doors-and-Panels_Images_Image032.jpg',10)", 'restats')

	import pstats
	p = pstats.Stats('restats')
	p.sort_stats('cumulative').print_stats(20)
	p.sort_stats('tottime').print_stats(20)
	
	# p.strip_dirs().sort_stats(-1).print_stats()