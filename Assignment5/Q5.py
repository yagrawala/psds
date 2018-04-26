import pandas as pd
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
# import sklearn

def read_data(filepath, columns):
	d = pd.read_csv(filepath, header=0, names=columns)
	return d

def q5_parta(filepath="DATAT/2016.csv"):
	d = read_data(filepath, ['Rk','Team','G','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS'])
	FG = d['FG%']
	TRB = d['TRB']
	PTS = d['PTS']
	TRB = TRB[:-1]
	FG = FG[:-1]
	PTS = PTS[:-1]

	X = np.zeros((len(FG),2))
	X[:,0] = FG[:]
	X[:,1] = TRB[:] 
	# X = np.hstack((np.array(FG),np.array(TRB)))
	Y = np.zeros((len(PTS),1))
	Y[:,0] = PTS[:] 


	# print(X)
	# print(Y)

	# print(X.shape)
	# print(Y.shape)
	beta = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X),X)), np.matmul(np.matrix.transpose(X), Y))
	print beta
	return beta

def q5_partb():
	d = read_data("DATAT/2016.csv", ['Rk','Team','G','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS'])
	FG = d['FG%']
	TRB = d['TRB']
	PTS = d['PTS']
	ORB = d['ORB']
	DRB = d['DRB']

	TRB = TRB[:-1]
	FG = FG[:-1]
	PTS = PTS[:-1]
	ORB = ORB[:-1]
	DRB = DRB[:-1]

	X = np.zeros((len(FG),4))
	X[:,0] = FG[:]
	X[:,1] = TRB[:] 
	X[:,2] = ORB[:]
	X[:,3] = DRB[:]
	# X = np.hstack((np.array(FG),np.array(TRB)))
	Y = np.zeros((len(PTS),1))
	Y[:,0] = PTS[:] 


	# print(X)
	# print(Y)

	# print(X.shape)
	# print(Y.shape)
	beta = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X),X)), np.matmul(np.matrix.transpose(X), Y))
	print beta
	return beta


def computeSSE(Y, Y_pred):
	error = np.matmul(np.matrix.transpose(Y-Y_pred), Y-Y_pred)
	sum_error = np.sum(error)
	return sum_error

def computeMAPE(Y, Y_pred):
	mape = 0
	for i in range(Y.shape[0]):
		mape = mape + abs(Y[i,0] - Y_pred[i,0])/Y[i,0]
	mape = mape*(100/Y.shape[0])
	return mape

def q5_parta_beta0(filepath="DATAT/2016.csv"):
	d = read_data(filepath, ['Rk','Team','G','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS'])
	FG = d['FG%']
	TRB = d['TRB']
	PTS = d['PTS']
	TRB = TRB[:-1]
	FG = FG[:-1]
	PTS = PTS[:-1]
	O = np.ones((len(FG),1))

	X = np.zeros((len(FG),3))
	X[:,0] = list(O)[:]
	X[:,1] = FG[:]
	X[:,2] = TRB[:] 

	# X = np.hstack((np.array(FG),np.array(TRB)))
	Y = np.zeros((len(PTS),1))
	Y[:,0] = PTS[:] 


	# print(X)
	# print(Y)

	# print(X.shape)
	# print(Y.shape)
	beta = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X),X)), np.matmul(np.matrix.transpose(X), Y))
	print beta
	return beta

def q5_partc(filepath1, filepath2):
	beta = q5_parta_beta0(filepath1)

	d = read_data(filepath2, ['Rk','Team','G','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS'])
	FG = d['FG%']
	TRB = d['TRB']
	PTS = d['PTS']
	TRB = TRB[:-1]
	FG = FG[:-1]
	PTS = PTS[:-1]
	O = np.ones((len(FG),1))

	X = np.zeros((len(FG),3))
	X[:,0] = list(O)[:]
	X[:,1] = FG[:]
	X[:,2] = TRB[:]
	Y = np.zeros((len(PTS),1))
	Y[:,0] = PTS[:]

	beta_array = np.ones((len(beta),1))
	beta_array = beta[:]
	Y_pred = np.matmul(X, beta_array)

	print(Y)
	print("")
	print(Y_pred)

	assert Y_pred.shape[0] == Y.shape[0]
	sse = computeSSE(Y, Y_pred)
	mape = computeMAPE(Y, Y_pred)

	return sse, mape, Y, Y_pred

def plotfig(X, Y, x_label, y_label):
	# plt.plot(X, Y, color='orange', linestyle='solid')
	plt.scatter(X,Y, verts=np.hstack((X,Y)))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

def q5_partd(filepath1, filepath2):
	sse, mape, Y, Y_pred = q5_partc(filepath1, filepath2)
	print("sse = " + str(sse))
	print("mape = " + str(mape))
	epsilon = Y-Y_pred
	plotfig(np.array(Y_pred), np.array(epsilon),'Y_pred','residuals')

def findFnHat(inputs):
    fnHat = []
    for i in inputs:
        count = 0
        for j in inputs:
            if j<i:
                count+=1
        fnHat.append(float(count)/len(inputs))
    return fnHat

def findPnHat(inputs):
    fnHat = []
    for i in inputs:
        count = 0
        for j in inputs:
            if j==i:
                count+=1
        fnHat.append(float(count)/len(inputs))
    return fnHat

def q5_parte(filepath1, filepath2):
	sse, mape, Y, Y_pred = q5_partc(filepath1, filepath2)
	print("sse = " + str(sse))
	print("mape = " + str(mape))
	epsilon = Y-Y_pred
	# fnhat = findFnHat(epsilon)
	# pnhat = findPnHat(epsilon)
	# plotfig(np.array(epsilon), np.array(fnhat),'residuals','fnhat')
	plt.hist(epsilon)
	plt.xlabel('epsilon')
	plt.ylabel('probability')
	plt.show()
	# plotfig(np.array(epsilon), np.array(pnhat),'residuals','pnhat')

if __name__ == "__main__":
	q5_parta()
	# q5_parta_beta0()

	# sse, mape, Y, Y_pred = q5_partc("DATAT/2016.csv", "DATAT/2017.csv")
	# print("sse = " + str(sse))
	# print("mape = " + str(mape))

	# sse, mape, Y, Y_pred = q5_partc("DATAT/2015_2016.csv", "DATAT/2017.csv")
	# print("sse = " + str(sse))
	# print("mape = " + str(mape))

	# sse, mape, Y, Y_pred = q5_partc("DATAT/2010_2016.csv", "DATAT/2017.csv")
	# print("sse = " + str(sse))
	# print("mape = " + str(mape))

	# q5_partd("DATAT/2016.csv", "DATAT/2017.csv")
	# q5_partd("DATAT/2015_2016.csv", "DATAT/2017.csv")
	# q5_partd("DATAT/2010_2016.csv", "DATAT/2017.csv")

	# q5_parte("DATAT/2016.csv", "DATAT/2017.csv")
	# q5_parte("DATAT/2015_2016.csv", "DATAT/2017.csv")
	# q5_parte("DATAT/2010_2016.csv", "DATAT/2017.csv")

