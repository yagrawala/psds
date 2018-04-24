import util
import numpy as np

x_vals = np.array([1.1, 2.2, 3.3, 4.4]).astype(np.float)
y_vals = np.array([5.5, 6.6, 7.7, 8.8]).astype(np.float)

# read files
pass



# create plots
# implementations in boolean flag to prevent interactive plots
if False:
	util.plot_normal(0, 1)
if True:
	X_range = np.array( range(0, len(x_vals)) )
	plot1   = util.getYPlotObj(x_vals, "line", "X1", "red")
	plot2   = util.getYPlotObj(y_vals, "scatter", "X2", "green")
	util.plot_graph(X_range, [plot1, plot2], showLegend=True)



# getStats
[x_mean, x_var, x_min, x_max] = util.getStats( x_vals )
[x_mean, x_var, x_min, x_max] = util.getStats( x_vals , printStats=True)



# tests
result = util.walds_test_2_population(x_vals, y_vals, thres=1.962)
result = util.paired_t_test(x_vals, y_vals, thres=1.962)
result = util.permutation_test(x_vals, y_vals, thres=1.962)



# time series
[average_error, errors, predictions] = util.make_predictions(x_vals, method="ewma", ewma_factor=0.5)
# [average_error, errors, predictions] = util.make_predictions(x_vals, method="seasonal", season_factor=2)
# [average_error, errors, predictions] = util.make_predictions(x_vals, method="ar", ar_factor=2)
