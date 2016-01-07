from bs4 import BeautifulSoup
from urlparse import urlparse,parse_qs
from py_bing_search import *
import cPickle
import csv
import time
import os
import hashlib
import urllib
from multiprocessing import Process, Pool, Lock, Manager,Queue
import multiprocessing
import timeit
import socket

socket.setdefaulttimeout(15)

class getImages(object):
	"""docstring for getImages"""
	# def __init__(self, arg):
	# 	super(getImages, self).__init__()
	# 	self.arg = arg
	categories = []
	verbose = True
	categories_cat_id_index = {}

	def readFile(self,file_name):
		f = open('../data/google_images/door.html', 'r')
		html_doc = f.read()
		soup = BeautifulSoup(html_doc, 'html.parser')
		#print soup.find_all('a.rg_l')
		for thumbnail in soup.find_all('a','rg_l'):
			href = thumbnail.get('href')
			o = urlparse(href)
			parsed_qs = parse_qs(o.query)
			img_url = parsed_qs['imgurl'][0]
			source = parsed_qs['imgrefurl'][0]
			print source
		#pass

	def getCategories(self):
		
		with open('./harvy/data/0_types.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			next(reader, None)  # skip the headers
			for row in reader:
				aux = {}
				aux['id'] = row[0]
				aux['name'] = row[1]
				aux['slug'] = row[2]
				self.categories.append(aux)
				self.categories_cat_id_index[aux['id']] = aux

	def createImageFolders(self):
		
		self.getCategories()

		for c in self.categories:
			path = './harvy/data/images/' + c['id'] +'_' + c['slug']
			if not os.path.exists(path):
				print path
    			os.makedirs(path)

	def downloadImagesParallel(self,urls):

		# Load the image in parallel using as many cpus as possible
		#spcpu = self.stop_position_per_cpu(len(im_paths)) # get number of cpu available
		num_cpus = multiprocessing.cpu_count()

		start_time = timeit.default_timer()
		manager = Manager()
		imlist = manager.list()
		im_index = manager.list()

		items_per_cpu = len(urls) / num_cpus
		# Start all the processes with evently distributed load
		p = []
		for i in xrange(0,num_cpus):

			start_i = i * items_per_cpu if i == 0 else i * items_per_cpu + 1
			stop_i = ((i+1) * items_per_cpu)

			print "start_i:{} stop_i:{}".format(start_i,stop_i)

			batch = urls[start_i:stop_i]
			#batch = 'lastro'
			#print len(batch)
			p.append(Process(target=self.downloadImagesProcess, args=([batch])))
			p[i].start()

		for i in xrange(0,num_cpus):
			p[i].join()

		# imlist = list(imlist)
		# im_index = list(im_index)
		
		if self.verbose:
			end_time = timeit.default_timer()
			elapsed_time = ((end_time - start_time))
			print "Elapsed time {}s loading images".format(elapsed_time)

		# return imlist,im_index
  #   	pass

  	def getImagesUrls(self):
  		self.getCategories()

		# There are 1194 categories
		#for j in range(4,len(self.categories)):

		urls = []
		for j in range(60,100):
			
			c = self.categories[j]
			path = './harvy/data/csv/' + c['id'] +'_' + c['slug'] + '.csv'

			if os.path.exists(path):
				with open(path, 'rb') as csvfile:
					reader = csv.reader(csvfile, delimiter=',', quotechar='"')
					next(reader, None)  # skip the headers
					i = 0 # use this to avoid duplicate images
					for row in reader:
						aux = {}
						aux['id'] = c['id']
						aux['slug'] = c['slug']
						aux['url'] = row[1]
						aux['j'] = j
						urls.append(aux)
		return urls

	def downloadImagesProcess(self,urls):
		
		i = 0 # use the i to avoid image that might have the same name but the content is diferent
		for u in urls:

			image_url = u['url']

			#generate a name for the image using sha224 so we won't have problems with encodings
			image_name = image_url.split("/")[-1]
			extension = os.path.splitext(image_name)[1]
			image_name_hashed = str(i) + hashlib.sha224(image_name).hexdigest() + extension


			file_path = './harvy/data/images/' + u['id'] + '_' + u['slug'] + '/' + image_name_hashed
			print file_path
			
			# If file is not in the file path, then download from the url
			if not os.path.exists(file_path):
				try:
					urllib.urlretrieve(image_url, file_path )
					print "j:{} i:{} image_url:{}".format(u['j'],i,image_url)
				except Exception, e:
					print e
			#else:
				#print "already downloaded"
			i += 1

	#Synchronous download. Use Parallel for faster download
	def downloadImages(self):
		self.getCategories()
		
		#for c in self.categories:
		for j in range(10,len(self.categories)):
			
			c = self.categories[j]
			path = './havy/data/csv/' + c['id'] +'_' + c['slug'] + '.csv'

			if os.path.exists(path):
				with open(path, 'rb') as csvfile:
					reader = csv.reader(csvfile, delimiter=',', quotechar='"')
					next(reader, None)  # skip the headers
					i = 0 # use this to avoid duplicate images
					for row in reader:
						#print c['id']
						image_url = row[1]
						image_name = image_url.split("/")[-1]
						extension = os.path.splitext(image_name)[1]
						image_name_hashed = str(i) + hashlib.sha224(image_name).hexdigest() + extension

						#print image_name_hashed
						file_path = './harvy/data/images/' + c['id'] + '_' + c['slug'] + '/' + image_name_hashed
						print file_path
						
						# If file is not in the file path, then download from the url
						if not os.path.exists(file_path):
							try:
								urllib.urlretrieve(image_url, file_path )
								print "j:{} i:{} image_url:{}".format(j,i,image_url)
							except Exception, e:
								print e
						i += 1

	def getImagesBing(self):
		#api_key = '3rnpY+p5FSX9k8FxMrJsR9KJKsrI5UzDeDe9y25RA1k'
		api_key = 'Tylgqmua/WWzy3M3tRA247brG0c9xA7gg04pVwciHpE'
		#api_key = 'euEUBWwX9h2ApzX7dAeO9NR6bd10vGPF92fOsB+WgUI'
		bing = PyBingSearch(api_key)		
		#result_list, next_uri = bing.search("loremdfadshfljkds jkdsa fhjklads fhalf hdjlsf adhslf adhsf ldsahfl ads", limit=50, format='json')
		print len(self.categories)
		for i in range(1193,len(self.categories)):
			c = self.categories[i]
			print c
			print i
			results = []
			for j in range(0,20):
				print j
				try:
					result_list, next_uri = bing.search(c['name'], limit=50, offset=50 * j, format='json')
					if next_uri != None:
						results += result_list
				except Exception, e:
					print e
					break
				
				if next_uri == None:
					break
				#time.sleep(1)
				print len(results)

			# output = open('../data/bing_res_aux.pkl', 'wb')
			# cPickle.dump(results, output,protocol=-1)
			# output.close()
				
			# pkl_file = open('../data/bing_res_aux.pkl', 'rb')
			# results = cPickle.load(pkl_file)
			#print results[-1].__dict__
			self.storeImagesCSV(c['id'], c['id'] +'_' + c['slug'], results)
			#break

	def getImagesBingMissingCategories(self):
		#api_key = '3rnpY+p5FSX9k8FxMrJsR9KJKsrI5UzDeDe9y25RA1k'
		api_key = 'Tylgqmua/WWzy3M3tRA247brG0c9xA7gg04pVwciHpE'
		#api_key = 'euEUBWwX9h2ApzX7dAeO9NR6bd10vGPF92fOsB+WgUI'
		bing = PyBingSearch(api_key)		
		#result_list, next_uri = bing.search("loremdfadshfljkds jkdsa fhjklads fhalf hdjlsf adhslf adhsf ldsahfl ads", limit=50, format='json')
		print len(self.categories)

		pkl_file = open('./harvy/data/missing_categories.pkl', 'wb')
		categories = cPickle.load(pkl_file)
		pkl_file.close()

		#for i in range(1193,len(self.categories)):
		for key in categories.keys():
			c = categories[key]
			#c = self.categories[i]
			print c
			print i
			results = []
			for j in range(0,20):
				print j
				try:
					result_list, next_uri = bing.search(c['name'], limit=50, offset=50 * j, format='json')
					if next_uri != None:
						results += result_list
				except Exception, e:
					print e
					break
				
				if next_uri == None:
					break
				#time.sleep(1)
				print len(results)

			self.storeImagesCSV(c['id'], c['id'] +'_' + c['slug'], results)



	def storeImagesCSV(self,cat_id, file_name, data):

		with open('./harvy/data/csv/' + file_name + '.csv' , 'wb') as csvfile:
			
			writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(['cat_id' , 'media_url' , 'source_url'])

			for item in data:
				try:
					writer.writerow([cat_id, item.media_url, item.source_url])
				except Exception, e:
					print e
				

	def pipeline(self):
		
		self.getCategories()
		self.createImageFolders()
		#self.getImagesBingMissingCategories()


if __name__ == '__main__':
	g = getImages()
	#urls = g.getImagesUrls()
	#g.downloadImagesParallel(urls)
	g.pipeline()
	#g.downloadImages()
	#g.createImageFolders()
	#g.getCategories()
	#g.getImagesBing()
	#gI.readFile('foor')
	#gI.getImagesBing()
		