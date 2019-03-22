import numpy as np 

from utils import simulate_x, nonlinear, simulate_params

from sklearn.preprocessing import StandardScaler




'''

Simulate 4 types of observed covariates: X_c are a set of covariates that influence both the treatment and outcome, also called confouders. X_iv are a set of covariates only influence treatment and not outcome, also called nstrumental variables. X_y are a set of covariates only influence the outcome and not treatment. X_n are a set of covariates influencing neither of treatment and outcome.

Accordingly, we define the number of covariates (dimension of columns) as p_c, p_iv, p_y, p_n

We fix the correlation between the covarites in each set to follow the same pattern with the base correlation rho. The correlation pattern is so that corr(X_j, X_k) = rho^(j-k). This is known as auto-regressive, AR(1).

We let the range of signals sizes be between r1 and r2.

We let the treatment be a binary treatment with probability of treated to be pr.

Let's agument some nonlinear functions of the columns in each covariate matrix:

f1(x1, x2) = exp(x1*x2/2)

f2(x1, x2) = x1/(1+exp(x2))

f3(x1, x2) = (x1*x2/10+2)^3

f4(x1, x2) = (x1+x2+3)^2

f5(x1, x2) = g(x1) * h(x2)

where

g(x) = -2 I(x < -1) - I(-1 < x < 0) + I(0 < x < 2)+ 3 I(x > 2)

h(x) = -5 I(x < 0) - 2 I(0 < x < 1) + 3 * I(x > 1)

and

f6(x1, x2) = g(x1) * h(x2)

where

g(x) = I(x > 0)

h(x) = I(x > 1)

We let "nonlinearity_portion" proportion of covariates in each matrix to be replaced by randomly selected nonlinearities. These nonlinearities are bivariate. We randomly select two of the covariates each time and implement the nonlinearities.

'''

def SimulateNoIntraction(True_TE, n, p_c, p_iv, p_y, p_n, rho=.5, corr="AR(1)", nonlinearity_portion=.10, r1=.1, r2=1., sigma=1.):

	X_c = simulate_x(n=n, p=p_c, rho=rho, mean=0., sd=1., corr="AR(1)")
	X_iv = simulate_x(n=n, p=p_iv, rho=rho, mean=0., sd=1., corr="AR(1)")
	X_y = simulate_x(n=n, p=p_y, rho=rho, mean=0., sd=1., corr="AR(1)")
	X_n = simulate_x(n=n, p=p_n, rho=rho, mean=0., sd=1., corr="AR(1)")

	X_c_latent = nonlinear(X_c, nonlinearity_portion=nonlinearity_portion)# nonlinearity_portion is the proportion of columns that are replaced by nonlinear functions.
	X_iv_latent = nonlinear(X_iv, nonlinearity_portion=nonlinearity_portion)
	X_y_latent = nonlinear(X_y, nonlinearity_portion=nonlinearity_portion)

	r = ["uniform", r1, r2]

	param_iv = simulate_params(p1=p_iv + p_c, r=r)
	xbeta_iv = np.dot(np.hstack((X_iv_latent, X_c_latent)), param_iv)
	pr = 1./(1. + np.exp(-xbeta_iv))
	A = np.random.binomial(1, pr, size=(n, 1))

	param_y = simulate_params(p1=p_y + p_c, r=r)

	xbeta_y = np.dot(np.hstack((A, np.hstack((X_y_latent, X_c_latent)))), np.vstack((True_TE, param_y))).reshape(-1, 1)

	y = xbeta_y + np.random.normal(size=(n, 1))

	index_c = np.arange(p_c)
	index_iv = np.arange(p_c, p_c + p_iv)

	x_ = np.concatenate([X_c, X_iv, X_y, X_n], axis=1)# Big data with all covarites (excluding th treatment)

	sdtz = StandardScaler()
	x = sdtz.fit_transform(x_)
	return(y, A, x) # Mean 0 and unit variance





def SimulateWithIntraction(True_TE, n, p_c, p_iv, p_y, p_n, rho=.5, corr="AR(1)", nonlinearity_portion=.10, interaction_portion=.1, r1=.1, r2=1., sigma=1.):

	X_c = simulate_x(n=n, p=p_c, rho=rho, mean=0., sd=1., corr="AR(1)")
	X_iv = simulate_x(n=n, p=p_iv, rho=rho, mean=0., sd=1., corr="AR(1)")
	X_y = simulate_x(n=n, p=p_y, rho=rho, mean=0., sd=1., corr="AR(1)")
	X_n = simulate_x(n=n, p=p_n, rho=rho, mean=0., sd=1., corr="AR(1)")

	X_c_latent = nonlinear(X_c, nonlinearity_portion=nonlinearity_portion)# nonlinearity_portion is the proportion of columns that are replaced by nonlinear functions.
	X_iv_latent = nonlinear(X_iv, nonlinearity_portion=nonlinearity_portion)
	X_y_latent = nonlinear(X_y, nonlinearity_portion=nonlinearity_portion)

	r = ["uniform", r1, r2]

	param_iv = simulate_params(p1=p_iv + p_c, r=r)
	xbeta_iv = np.dot(np.hstack((X_iv_latent, X_c_latent)), param_iv)
	pr = 1./(1. + np.exp(-xbeta_iv))
	A = np.random.binomial(1, pr, size=(n, 1))

	param_y = simulate_params(p1=p_y + p_c, r=r)

	xbeta_y = np.dot(np.hstack((A, np.hstack((X_y_latent, X_c_latent)))), np.vstack((True_TE, param_y))).reshape(-1, 1)

	# number of interaction terms
	n_interactions = int(nonlinearity_portion*p_c)
	x_interactions = A * X_c[:, np.random.choice(p_c, n_interactions, replace=False)]
	param_interactions = simulate_params(p1=n_interactions, r=r)

	interactions = np.dot(x_interactions, param_interactions).reshape(-1, 1)

	y = xbeta_y + interactions + sigma * np.random.normal(size=(n, 1))

	index_c = np.arange(p_c)
	index_iv = np.arange(p_c, p_c + p_iv)

	x_ = np.concatenate([X_c, X_iv, X_y, X_n], axis=1)# Big data with all covarites (excluding th treatment)

	sdtz = StandardScaler()
	x = sdtz.fit_transform(x_)
	return(y, A, x) # Mean 0 and unit variance


