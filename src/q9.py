from __future__ import division
import math
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def read_values( filename ):
	lines = [line.rstrip('\n') for line in open(filename)]
	counter = 0
	vals = []
	for line in lines:
		vals.append( line.split(",") )
		counter += 1
	return np.array(vals).astype(np.float)

def calculate_posterior(data, prior, se_square):
	deno      = prior[1] + se_square
	mean_num  = prior[1] * np.mean( data ) + prior[0] * se_square
	var_num   = prior[1] * se_square
	post_mean = mean_num / deno
	post_var  = var_num / deno
	return [post_mean, post_var]

def plot_normal( posteriors ):
	for i in range(0, len(posteriors) ):
		mu = posteriors[i][0]
		sigma = math.sqrt( posteriors[i][1] )
		x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
		plt.plot(x, mlab.normpdf(x, mu, sigma), label=str(i) )
	plt.show()

# read the values from the file
sigma = 100
vals = read_values( "q9_sigma"+str(sigma)+".dat" )

# initialize variables
prior_mean = 0
prior_var  = 1
posteriors = []
se_square  = ( sigma**2 ) / vals.shape[1]

# calculate the posterior
for i in range(0, vals.shape[0] ):
	[prior_mean, prior_var] = calculate_posterior( vals[i], (prior_mean, prior_var), se_square)
	posteriors.append( (prior_mean, prior_var) )

plot_normal( posteriors )
print( posteriors )
