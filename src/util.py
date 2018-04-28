########################################################
# File Name - util.py
# Type      - Module
# Purpose   - Implementation of common code for -
#			  1. reusability
#			  2. uniformity in analysis
########################################################


##############################
## required libraries here	##
##############################
from __future__ import division
import math
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression


##################################
## methods to manipulate file  	##
##################################
# reads a file with horizontal data
# eg - "1,2,3,4"
# returns numpy array
def read_horizontal_values(filename, delimiter=","):
	lines = [line.rstrip('\n') for line in open(filename)]
	vals = lines[0].split( delimiter )
	return np.array(vals).astype(np.float)

def read_2D_values(filename, delimiter=",", npWanted=False):
    lines = [line.rstrip('\n') for line in open(filename)]
    vals = []
    for line in lines:
        vals.append( line.split( delimiter ) )
    if npWanted:
        vals = np.array(vals).astype(np.float)
    return vals



##################################
## methods to create plots  	##
##################################
def plot_normal(mu, sigma, label="data"):
	x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
	plt.plot(x, mlab.normpdf(x, mu, sigma), label=label )
	plt.show()

def getYPlotObj(values, plot_type, label=False, color="blue"):
    obj =  {
        'values': np.array(values),
        'type'  : plot_type,
        'color' : color
    }
    if label:
        obj['label'] = label
    return obj

def plot_graph(X_range, Y_vals, X_label="X-Axis", Y_label="Y-Axis", showLegend=False):
    X_range = np.array( X_range )
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(0, len(Y_vals)):
        plot_type = Y_vals[i]['type']
        y_values  = np.array( Y_vals[i]['values'] )
        color     = Y_vals[i]['color']
        if "label" in Y_vals[i].keys():
            label = Y_vals[i]['label']
        else:
            label = ""
        if plot_type == "line":
            ax.plot(	X_range, 
                        y_values,
                        color=color,
                        label=label,
                        linestyle='solid'
            )
        if plot_type == "scatter":
            ax.scatter(	X_range,
                            y_values,
                            color=color,
                            label=label,
                            s=10,
                            alpha=1
            )
    plt.xlabel( X_label )
    plt.ylabel( Y_label )
    if showLegend:
        fontP = FontProperties()
        fontP.set_size('small')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)
    plt.show()



######################################
## get generic stats for some data 	##
######################################
# vals should be a numeric array
def getMean( vals ):
	return np.mean( vals )

# vals should be a numeric array
def getVar( vals ):
	return np.var( vals )

# vals should be a numeric array
def getMin( vals ):
	return np.amin( vals )

# vals should be a numeric array
def getMax( vals ):
	return np.amax( np.array(vals).astype(np.float) )

# vals should be a numeric array
def getStats( vals , printStats=False):
	mean = getMean( vals )
	var  = getVar( vals )
	mini = getMin( vals )
	maxi = getMax( vals )
	if printStats:
		print("\n======= Stats ========")
		print( "mean = " + str( round(mean, 4) ) )
		print( "var  = " + str( round(var, 4) ) )
		print( "min  = " + str(mini) )
		print( "max  = " + str(maxi) )
		print("======================\n")
	return [mean, var, mini, maxi]

# SSE = sum squared error
def getSSE(Y, Y_pred):
	SSE_val = 0
	for i in range(0, len(Y) ):
		epis_val = Y[i] - Y_pred[i]
		SSE_val += epis_val**2
	SSE = SSE_val.tolist()[0]
	return SSE

# MAPE = mean average percentage error
def getMAPE(Y, Y_pred):
	MAPE_val = 0
	count = 0
	for i in range(0,len(Y)):
		if Y[i] != 0:
			map_val = abs(Y[i] - Y_pred[i])/abs(Y[i])
			MAPE_val += map_val
			count += 1
	MAPE = (MAPE_val*100/count).tolist()[0]
	return MAPE



##################################
## all tests implemented here	##
##################################
# x_data, y_data should be numeric arrays of same length
# returns boolean, based on whether threshold matches
def walds_test_2_population(x_data, y_data, thres):
	print("\n==== Wald's Test for 2 Population ====")

	# first check the lengths match or not
	if len( x_data ) != len( y_data ):
		print("Length of the 2 populations do not match.")
		return

	# run the test
	N = len( x_data )
	[x_mean, x_var] = getStats( x_data )[:2]
	[y_mean, y_var] = getStats( y_data )[:2]
	se = math.sqrt( (x_var + y_var) / N )
	w = abs(x_mean - y_mean) / se
	result = ( w < thres )

	# print info
	print( "se     = " + str(se) )
	print( "w      = " + str(w) )
	print( "thres  = " + str(thres) )
	if result:
		print("result = Passed")
	else:
		print("result = Failed")
	print("======================================\n")
	return result

# x_data, y_data should be numeric arrays of same length
# returns boolean, based on whether threshold matches
def paired_t_test(x_data, y_data, thres):
	print("\n==== Paired-t Test ====")

	# first check the lengths match or not
	if len( x_data ) != len( y_data ):
		print("Length of the 2 populations do not match.")
		return

	# run the test
	diff_values = x_data - y_data
	[diff_mean, diff_var] = getStats( diff_values )[:2]
	deno = math.sqrt( diff_var / len(x_data) )
	w = abs( diff_mean ) / deno
	result = ( w < thres )

	# print info
	print( "deno   = " + str(deno) )
	print( "w      = " + str(w) )
	print( "thres  = " + str(thres) )
	if result:
		print("result = Passed")
	else:
		print("result = Failed")
	print("=========================\n")
	return result

# x_data, y_data should be numeric arrays of same length
# returns boolean, based on whether threshold matches
def permutation_test(x_data, y_data, thres, permutations=100):
	print("\n==== Permutation Test ====")

	# first check the lengths match or not
	if len( x_data ) != len( y_data ):
		print("Length of the 2 populations do not match.")
		return

	# run the test
	N = len( x_data )
	[x_mean, x_var] = getStats( x_data )[:2]
	[y_mean, y_var] = getStats( y_data )[:2]
	t_obs = abs( x_mean - y_mean )
	se   = math.sqrt( (x_var + y_var) / N )
	w = t_obs / se

	# run permutations
	united = np.append( x_data, y_data )
	t_perm = []
	count  = 0
	for i in range(0, permutations):
		perm   = np.random.permutation( united )
		x_perm = perm[:N]
		y_perm = perm[N:]
		x_mean = getMean( x_perm )
		y_mean = getMean( y_perm )
		t_new  = abs( x_mean - y_mean )
		t_perm.append( t_new )
		if t_new > t_obs:
			count += 1
	p_value = count / permutations
	result = (p_value > thres)

	# print info
	print( "t_obs  = " + str(t_obs) )
	print( "se     = " + str(se) )
	print( "w      = " + str(w) )
	print( "p      = " + str(p_value) )
	if result:
		print("result = Passed")
	else:
		print("result = Failed")
	print("==========================\n")
	return result



######################################
## all time series implemented here	##
######################################
# implementation of emwa method of prediction
def ewma(data, alpha):
	conj_alpha = 1 - alpha
	prediction = alpha*data[-1]
	for i in range(len(data)-1, -1, -1):
		# print( i+1 )
		prediction += conj_alpha*data[i]
		conj_alpha *= conj_alpha
	return prediction

class autoRegression:
	betas = []

	def __init__(self, data, interval, threshold):
		# create data from given interval
		x_s = []
		y_s = []
		for i in range(0, threshold-interval):
			x = [1] + data[i:i+interval]
			y = data[i+interval]
			x_s.append( np.array( x ) )
			y_s.append( y )

		X = np.array( x_s )#.reshape( (threshold, interval+1) )
		Y = np.array( y_s ).reshape( (len(y_s), 1) )
		lm = LinearRegression()
		self.betas = lm.fit(X ,Y)

	def predict(self, features):
		features = [1] + features
		features = np.array( features ).reshape(1, len(features) )
		return self.betas.predict( features )[0][0]

# used for making predictions, by the specified method and paramter for that methods
def make_predictions(data, method="ewma", ewma_factor=0.5, season_factor=144, ar_factor=144, test_start_idx=576):
	errors = []
	predictions = []

	if method == "ar":
		model = autoRegression(data, ar_factor, test_start_idx)

	for i in range(test_start_idx, len(data)):

		#######################################################
		# handle all methods here
		if method == "ewma":
			train_data = data[:i]
			prediction = ewma(train_data, ewma_factor)

		elif method == "seasonal":
			prediction = data[i-season_factor+1]

		elif method == "ar":
			prediction = model.predict( data[i-ar_factor:i] )

		else:
			print("invalid method passed."); return
		#######################################################

		error = abs(prediction - data[i])
		errors.append( error )
		predictions.append( prediction )

	errors = np.array(errors).astype(np.float)
	average_error = np.mean( errors )

	print("average_error=", average_error)
	return [average_error, errors, predictions]


