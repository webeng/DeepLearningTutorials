import theano.tensor as T
from theano import function
from theano.ifelse import ifelse
import theano, time, numpy
from theano import shared
rng = numpy.random

state = shared(float(0))

x = T.dscalar('x')
y = T.dscalar('y')
#x = T.scalar(dtype= state.type)
#y = T.scalar(dtype= state.type)
z = x + y


#state = 1

#f = function([x, y], z)
f_updates = function([x, y], z , updates=[(state, state + x + y)])

# print f_updates(1,2)
# z_switch = T.switch(T.lt(x,y) , T.pow(x,2) + y , x + y)
# f_switch = function([x,y],z_switch)
# print f_switch(4,3)

# for i in xrange(1,10):
# 	f_updates(1,0)

# print state.get_value()

R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

R = numpy.array(R).astype(theano.config.floatX)
tR = theano.shared(R.astype(theano.config.floatX),name="R")
print type(tR)
ncols = len(R[0])
nrows = len(R)
#print 
#theano.printing.debugprint(tR.shape) 
#print
row_values = T.dvector('row_values')
column_values = T.dvector('column_values')
row = T.dscalar('row')

total_squared_sum = shared(float(0))
#sq_sum = pow(row_values.sum(),2) + 
dot_product = row + T.dot(row_values,row_values)
f_test = function([row,row_values], dot_product , updates=[(total_squared_sum, total_squared_sum + dot_product)])

for row in xrange(0,nrows):
    f_test(row,R[row,:])

print total_squared_sum.get_value()


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T

    # p_row = theano.shared(rng.random((1,4)).astype(theano.config.floatX))
    # q_col = theano.shared(rng.random((1,5)).astype(theano.config.floatX))

    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
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
    return P, Q.T

R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

R = numpy.array(R)

N = len(R)
M = len(R[0])
K = 2

# P = theano.shared(
#     numpy.asarray(
#         numpy.random.rand(N,K),
#         dtype=theano.config.floatX
#     ),
#     borrow=True
# )

# Q = theano.shared(
#     numpy.asarray(
#         numpy.random.rand(M,K),
#         dtype=theano.config.floatX
#     ),
#     borrow=True
# )

#print Q

# P = numpy.random.rand(N,K)
# Q = numpy.random.rand(M,K)

# nP, nQ = matrix_factorization(R, P, Q, K)
# nR = numpy.dot(nP, nQ.T)