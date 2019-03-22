


import numpy as np


from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


utils = importr("twang")
ro.r('library("twang")')

def IPW(y, A, x):


	nr,nc = x.shape
	x_r = ro.r.matrix(x, nrow=nr, ncol=nc)

	ro.r('A = c{}'.format(tuple(A.flatten())))

	ro.r('W = c{}'.format(tuple(x_r)))
	ro.r("W = matrix(W, nrow={}, ncol={})".format(nr, nc))

	ro.r('y = c{}'.format(tuple(y.flatten())))

	ro.r('data_r = data.frame(cbind(A, W))')

	formula_x = ""
	for x in ro.r('names(data_r)')[1:]:
	    formula_x += "+" + x

	ps_r_results = ro.r('ps_r <- ps(formula=A ~ {}, data=data_r, n.trees=200, shrinkage=.01)'.format(formula_x[1:]))

	gbm_ps = np.array([a[0] for a in ro.r('ps_r$ps')])

	A1, A0 = ps_gbm, 1-ps_gbm

	w1, w0 = 1/A1, 1/A0

	mu1 = np.sum(y * A * w1)/np.sum(A * w1)
	mu0 = np.sum(y * (1-A) * w0)/np.sum((1-A) * w0)

	TE_gbm_ps = mu1 - mu0
	return(TE_gbm_ps)