import theano.tensor as T
from theano import function
from theano.ifelse import ifelse
import theano, time, numpy
from theano import shared
rng = numpy.random
from datetime import datetime
import cPickle
theano.config.floatX = 'float32'
# state = shared(float(0))

# x = T.dscalar('x')
# y = T.dscalar('y')
# z = x + y

#f = function([x, y], z)
# f_updates = function([x, y], z , updates=[(state, state + x + y)])

# print f_updates(1,2)
# z_switch = T.switch(T.lt(x,y) , T.pow(x,2) + y , x + y)
# f_switch = function([x,y],z_switch)
# print f_switch(4,3)

# for i in xrange(1,10):
# 	f_updates(1,0)

# R = [
# 	 [5,3,0,1],
# 	 [4,0,0,1],
# 	 [1,1,0,5],
# 	 [1,0,0,4],
# 	 [0,1,5,4],
# 	]

# R = numpy.array(R).astype(theano.config.floatX)
# tR = theano.shared(R.astype(theano.config.floatX),name="R")

# ncols = len(R[0])
# nrows = len(R)

# row_values = T.dvector('row_values')
# column_values = T.dvector('column_values')
# row = T.dscalar('row')

# total_squared_sum = shared(float(0))
#sq_sum = pow(row_values.sum(),2) + 
# dot_product = row + T.dot(row_values,row_values)
# f_test = function([row,row_values], dot_product , updates=[(total_squared_sum, total_squared_sum + dot_product)])

# for row in xrange(0,nrows):
# 	f_test(row,R[row,:])

#print total_squared_sum.get_value()
def theano_matrix_factorization(steps=20000, alpha=0.0002, beta=0.02):

	R = [
	 [5,3,0,1],
	 [4,0,0,1],
	 [1,1,0,5],
	 [1,0,0,4],
	 [0,1,5,4],
	]

	R = numpy.array(R)

	#pkl_file = open( '../data/R.pkl', 'rb')
	pkl_file = open( '/home/ubuntu/R.pkl', 'rb')
	R = cPickle.load(pkl_file)

	N = len(R)
	M = len(R[0])
	K = 2

	P = theano.shared(
	numpy.asarray(
		numpy.random.rand(N,K),
		dtype=theano.config.floatX
	),
	borrow=True
	)

	Q = theano.shared(
	numpy.asarray(
		numpy.random.rand(M,K).T,
		dtype=theano.config.floatX
	),
	borrow=True
	)
	#Q = Q.T

	t_alpha = T.fscalar('alpha')
	t_beta = T.fscalar('beta')

	# t_alpha = T.dscalar('alpha')
	# t_beta = T.dscalar('beta')

	A = R.copy()
	A[ A > 0 ] = 1

	A = A.astype(theano.config.floatX)
	R = R.astype(theano.config.floatX)
	
	E = numpy.asarray(
		numpy.random.rand(len(R),len(R[0])),
		dtype=theano.config.floatX
	)

	# Let's try to do automatic gradient
	# tPCost = pow(E, 2).sum() + (beta/2) * (pow(P.get_value(),2).sum() + pow(Q.get_value(),2).sum())
	# tQCost = pow(E, 2).sum() + (beta/2) * (pow(P.get_value(),2).sum() + pow(Q.get_value(),2).sum())
	# cost = T.sum(T.pow(E,2)) + (beta/2) * (T.sum(T.pow(P,2)) + T.sum(T.pow(Q,2)))
	# grads = T.grad(cost, [P,Q,E])
	
	#AUX = numpy.dot(P,Q) * A
	#E = R - AUX
	AUX = T.dot(P,Q) * A
	E = R - AUX
	#print E
	train = theano.function(
			inputs=[t_alpha,t_beta],
			outputs=E,
			updates=[( P,  P + t_alpha * (2 * T.dot(E,Q.T) - t_beta * P) ) , (Q, Q + t_alpha * (2 * T.dot(P.T,E) - t_beta * Q) )],
			name="train")

	#print train(np.asarray(gamma,dtype=theano.config.floatX),np.asarray(l,dtype=theano.config.floatX));

	for step in xrange(steps):
		E = train(numpy.asarray(alpha,dtype=theano.config.floatX),numpy.asarray(beta,dtype=theano.config.floatX))
		
		# Let's calculate the error every 100 iteration so it's more effective
		if (step % 100) == 0:
			e = 0
			e = pow(E, 2).sum() + (beta/2) * (pow(P.get_value(),2).sum() + pow(Q.get_value(),2).sum())
			print "step: {} e: {}".format(step,e)
		
		#e = e + T.sum(T.pow(E, 2)) + (beta/2) * (T.sum(T.pow(P,2)) + T.sum(T.pow(Q,2)))
		#e = T.sum(T.pow(E, 2)) + (beta/2) * (T.sum(T.pow(P,2)) + T.sum(T.pow(Q,2)))
		#print "step: {}".format(step)

		# if e < 0.001:
		# 	break
	print "Min e: {}".format(e)
	return P.get_value(), Q.get_value().T

def matrix_factorization_vectorised(R, P, Q, K, steps=50000, alpha=0.0002, beta=0.02):
	Q = Q.T

	A = R.copy()
	A[ A > 0 ] = 1

	for step in xrange(steps):
		#print "Step: {} matrix_factorization_vectorised".format(step)
		# Calculate the current cost. We need an auxiliar matrix to not take into account the values of R that are z. We basically need it for the step if R[i][j] > 0: in the original algo
		AUX = numpy.dot(P,Q) * A
		E = R - AUX

		#E = R - numpy.dot(P,Q) # Original computation
		P = P + alpha * (2 * numpy.dot(E,Q.T) - beta * P)
		Q = Q + alpha * (2 * numpy.dot(P.T,E) - beta * Q)
		
		# print Q.shape
		# print P.shape        
		# print E.shape

		# for i in xrange(len(R)):
		#     for j in xrange(len(R[i])):
		#         if R[i][j] > 0:
		#             eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
		#             for k in xrange(K):
		#                 P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
		#                 Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

		# Let's calculate the error for the current step

		if (step % 100) == 0:
			e = 0
			e = e + pow(E, 2).sum() + (beta/2) * (pow(P,2).sum() + pow(Q,2).sum())
			print "step: {} e: {}".format(step,e)

		# e = 0
		# e = e + pow(E, 2).sum() + (beta/2) * (pow(P,2).sum() + pow(Q,2).sum())

		#eR = numpy.dot(P,Q)
		# e = 0
		# for i in xrange(len(R)):
		#     for j in xrange(len(R[i])):
		#         if R[i][j] > 0:
		#             e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
		#             for k in xrange(K):
		#                 e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
			break

		#print "e: {}".format(e)

		#print " matrix_factorization_vectorised e: {}".format(e)
	print "Min e: {}".format(e)
	return P, Q.T

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
	Q = Q.T
	
	for step in xrange(steps):
		#E = numpy.zeros((len(R),len(R[0])))
		# print "Step: {} matrix_factorization".format(step)
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
					#E[i][j] = eij
					#print "i: {} j: {} e: {}".format(i,j,eij)
					for k in xrange(K):
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		# print "P matrix_factorization"
		# print P
		# print "Q matrix_factorization"
		# print Q
		# print E
		eR = numpy.dot(P,Q)
		e = 0
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
					for k in xrange(K):
						e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
			break
		
		#print " matrix_factorization e: {}".format(e)
		#break
	print "Min e: {}".format(e)
	return P, Q.T

R = [
	 [5,3,0,1],
	 [4,0,0,1],
	 [1,1,0,5],
	 [1,0,0,4],
	 [0,1,5,4],
	]

R = numpy.array(R)

# pkl_file = open( '../data/R.pkl', 'rb')
# R = cPickle.load(pkl_file)

N = len(R)
M = len(R[0])
K = 2

P = theano.shared(
	numpy.asarray(
		numpy.random.rand(N,K),
		dtype=theano.config.floatX
	),
	borrow=True
)

Q = theano.shared(
	numpy.asarray(
		numpy.random.rand(M,K),
		dtype=theano.config.floatX
	),
	borrow=True
)

# Initialize P and Q
average_non_blank = R[R > 0].mean()
ini_values = numpy.sqrt(average_non_blank / K)
c = 1
perturbationP = numpy.random.uniform(-c,c,size=(N,K))
perturbationQ = numpy.random.uniform(-c,c,size=(M,K))

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

P.fill(ini_values)
Q.fill(ini_values)

P = P + perturbationP
Q = Q + perturbationQ

#print P

#print P

# print "Loop implementation" 
# startTime = datetime.now()
# nP, nQ = matrix_factorization(R, P, Q, K)
# print datetime.now() - startTime

print "Vectorised implementation"
startTime = datetime.now()
nP, nQ = matrix_factorization_vectorised(R, P, Q, K)
print datetime.now() - startTime

# print "GPU implementation" 
# startTime = datetime.now()
# nP, nQ = theano_matrix_factorization()
# print datetime.now() - startTime

# nR = numpy.dot(nP, nQ.T)
# print nR

#print datetime.now() - startTime