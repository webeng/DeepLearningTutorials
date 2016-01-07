from __future__ import division
from sklearn import svm, linear_model

X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[45],[70],[146],[170],[1246],[1296],[1746],[1821],[1996],[2496],[2746],[2996],[3496],[4496]]
#X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],[31],[32],[33],[34],[35],[36],[37],[38],[39],[40],[41],[42],[43],[44],[45],[46],[47],[48],[49],[50],[51],[52],[53],[54],[55],[56],[57],[58],[59],[60],[61],[62],[63],[64],[64],[65],[66],[67],[68],[69],[70]]

#X = [1,2,3,4,5,6,7]
#X = [[0, 0], [2, 2]]
y = [19111,2132,2088,1865,1854,1552,1378,1279,1239,1234,1051,1050,969,967,879,775,717,714,668,661,480,300,270,180,150,13,12,6,6,5,3,3,2,2,1]
#y = [19111,2132,2088,1865,1854,1552,1378,1279,1239,1234,1051,1050,969,967,879,775,717,714,668,661,480,470,460,450,440,430,420,410,400,390,380,370,360,350,345,340,335,330,325,320,315,310,305,302,300,299,298,297,296,295,294,293,292,291,290,289,288,287,286,285,284,283,282,281,280,279,278,277,276,275,270]

total_restaurants = 0
for i in xrange(0,len(X) - 1):
	if ((X[i][0] - i) == 1):
		print y[i]
		#total_restaurants += y[i]
		total_restaurants += y[i]
	else:
		diff = X[i][0] - i
		diff_values = y[i - 1] - y[i]
		decrease_per_step = diff_values / diff
		top_y = y[i]
		bottom_y = y[i - 1]

		prev = bottom_y
		for i in xrange(0,diff):
			dec = prev - decrease_per_step
			total_restaurants += dec
			prev = dec
		print "diff:{} diff_values:{} decrease_per_step:{} top_y:{} bottom_y:{}".format(diff,diff_values,decrease_per_step, top_y,bottom_y)

print "There are {} restaurants in Tripadvisor.".format(total_restaurants)

# #y = [19111,2132,2088,1865,1854,1552,1378,1279,1239,1234,1051,1050,969,967,879,775,717,714,668,661,480,300,270,180,150,13,12,6,6,5,3,3,2,2,1]
# #y = [19111,2132,2088,1865,1854,1552,1378]
# # print len(X)
# # print len(y)
# #clf = svm.SVR(kernel='poly')
# #clf = svm.SVR(kernel='linear')
# clf = svm.SVR(kernel='linear')
# #clf = svm.SVR(kernel='poly', degree=2)
# #clf = linear_model.SGDRegressor()
# clf.fit(X, y) 
# #print clf.predict([10])

# total_restaurants = sum(y[0:21])
# print total_restaurants
# #for i in xrange(22,5000):
# total_restaurants = 0
# for i in xrange(0,2000):
# 	print i
# 	print clf.predict(i)[0]
# 	total_restaurants = total_restaurants + clf.predict(i)[0]

# print "There are {} restaurants in Tripadvisor.".format(total_restaurants)
# #3736806
# #print clf.predict([4973])
# print clf.predict(4496)
# print clf.predict(100)