import math
import numpy as np

def read_values( filename ):
	lines = [line.rstrip('\n') for line in open(filename)]
	vals = lines[0].split(",")
	return np.array(vals).astype(np.float)

def getStats( vals ):
	mean = np.mean( vals )
	var  = np.var( vals )
	mini = np.amin( vals )
	maxi = np.amax( vals )
	return [mean, var, mini, maxi]

def walds_test(x_mean, x_var, y_mean, y_var, values, thres):
	print("\n==== Wald's Test ====")

	se = math.sqrt( (x_var + y_var) / values )
	print( "se = " + str(se) )

	w = abs(x_mean - y_mean) / se
	print( "w = " + str(w) )

	return w < thres

# def welchs_test(x_mean, x_var, y_mean, y_var, values, thres):
# 	print("\n==== Welch's Test ====")

# 	deno = math.sqrt( (x_var + y_var) / values )
# 	print( "deno = " + str(deno) )

# 	w = abs(x_mean - y_mean) / deno
# 	print( "w = " + str(w) )

# 	return w < thres

def paired_t_test(x_vals, y_vals, thres):
	print("\n==== Paired-t Test ====")

	diff_values = x_vals - y_vals
	diff_mean = np.mean( diff_values )
	diff_var  = np.var( diff_values )

	deno = math.sqrt( diff_var / len(x_vals) )
	print( "deno = " + str(deno) )

	w = abs( diff_mean ) / deno
	print( "w = " + str(w) )

	return w < thres

# read the values from the file
x_vals = read_values( "q6_X.dat" )
y_vals = read_values( "q6_Y.dat" )

# calculate the statistics for each value
[x_mean, x_var, x_min, x_max] = getStats( x_vals )
[y_mean, y_var, y_min, y_max] = getStats( y_vals )

# print them
print("==========================")
print( "x_mean = " + str(x_mean) )
print( "x_var  = " + str(x_var) )
print( "x_min  = " + str(x_min) )
print( "x_max  = " + str(x_max) )
print("==========================")
print( "y_mean = " + str(y_mean) )
print( "y_var  = " + str(y_var) )
print( "y_min  = " + str(y_min) )
print( "y_max  = " + str(y_max) )
print("==========================")

# run walds test
result = walds_test(x_mean, x_var, y_mean, y_var, len(x_vals), thres=1.962)
print( "Wald's Result = " + str(result) )

# run welchs test
# result = welchs_test(x_mean, x_var, y_mean, y_var, len(x_vals), thres=1.962)
# print( "Welch's Result = " + str(result) )

# run paired_t test
result = paired_t_test(x_vals, y_vals, thres=1.962)
print( "Paired-t Result = " + str(result) )

print("\n")
