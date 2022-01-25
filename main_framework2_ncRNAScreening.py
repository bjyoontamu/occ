import os
from re import A
import time
import numpy as np
from numpy.random import f
import pandas as pd
import scipy

from numpy.linalg import inv
from sklearn.mixture import GaussianMixture
from scipy import integrate
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds, fsolve, brentq
from scipy.stats import mvn, wishart, invwishart
from itertools import chain, combinations, permutations

from datetime import datetime
import pickle

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def mainScreeningNcRNA():
	isQuickTest = True
	prinprintOutput = True
	algorithms = ['proposedLambda', 'baseline']

	evaluateOCC_ncRNAScreening(isQuickTest, algorithms, "EM", prinprintOutput)

def evaluateOCC_ncRNAScreening(isQuickTest, algorithms, densityEstimation = "MLE", prinprintOutput = False):
	alpha = 0.8
	metrics = np.array(['CPC2', 'CPAT', 'PLEK', 'LncFinder'])

	df = pd.read_pickle("./data/v38._samples_score")
	if densityEstimation == "MLE":
		# GMM - MLE
		df_n = df[df['Label'] != 'ncRNA']
		df_p = df[df['Label'] == 'ncRNA']

		pi_n = len(df_n)/len(df)
		# pi_n = 0
		mu_n = np.mean(df_n[metrics].values, axis=0)
		sigma_n = np.cov(df_n[metrics].values.T, bias=False)  

		pi_p = len(df_p)/len(df)
		# pi_p = 1
		mu_p = np.mean(df_p[metrics].values, axis=0)
		sigma_p = np.cov(df_p[metrics].values.T, bias=False) 

		pi = np.array([pi_n, pi_p])	
	else:
		# GMM - EM
		df_p = df[df['Label'] == 'ncRNA']
		# x_train = df[metrics].values
		x_train	= df[metrics].sample(frac = 0.04, random_state=1).values
		# x_train	= df[metrics].sample(frac = 1, random_state=1).values

		# fit a Gaussian Mixture Model with two components
		modelDistribution = GaussianMixture(n_components=2, covariance_type='full').fit(x_train)

		pi = modelDistribution.weights_
		mu_n = modelDistribution.means_[0]
		mu_p = modelDistribution.means_[1]
		sigma_n = modelDistribution.covariances_[0]
		sigma_p = modelDistribution.covariances_[1]

	defaultCost = np.array([0.002526541940798, 0.0027336052908625, 0.0831765095718637, 2.495623060544432])*1000

	X0 = len(df)
	lambdaLast = -0.2
	GT = len(df[df[metrics[-1]] > lambdaLast])

	stageOrder = list(range(len(metrics)))
	allStageSetups = list(chain.from_iterable(combinations(stageOrder, r) for r in range(len(stageOrder)+1)))
	stageSetups = list()
	for i in range(len(allStageSetups)):
		stageSetup = allStageSetups[i]
		if (len(stageSetup) > 1) and (stageSetup[-1] == stageOrder[-1]):
			if len(stageSetup) > 2:
				allPermutations = list(permutations(stageSetup))
				for permutation in allPermutations:
					if (permutation[-1] == stageOrder[-1]):
						stageSetups.append(permutation)
			else:
				stageSetups.append(stageSetup)

	data = {}
	data['costPerStage'] = defaultCost
	data['numberOfSamples'] = X0
	data['GT'] = GT
	data['PS'] = len(df_p)
	i = 0
	for stageSetup in stageSetups: 
		if len(stageSetup) != 4:
			continue
		stageSetup = np.array(stageSetup)

		reorderedSigma_n = sigma_n
		reorderedSigma_n = reorderedSigma_n[:, stageSetup][stageSetup]
		reorderedMu_n = np.array(mu_n)
		reorderedMu_n = reorderedMu_n[stageSetup]

		reorderedSigma_p = sigma_p
		reorderedSigma_p = reorderedSigma_p[:, stageSetup][stageSetup]
		reorderedMu_p = np.array(mu_p)
		reorderedMu_p = reorderedMu_p[stageSetup]

		reorderedMu = np.array([reorderedMu_n, reorderedMu_p])
		reorderedSigma = np.array([reorderedSigma_n, reorderedSigma_p])

		reorderedC = np.array(defaultCost)
		reorderedC = reorderedC[stageSetup]

		listOutput = list() 
		dicOutput = runAlgorithms(isQuickTest, df, metrics[stageSetup], algorithms, pi, reorderedMu, reorderedSigma, X0, reorderedC, lambdaLast, alpha, GT, True, stageSetup, prinprintOutput)
		listOutput.append(dicOutput)
		outAlgorithms = dicOutput['algorithms']
		standardMethod = 0
		for i in range(len(outAlgorithms)):
			if 'proposed' in outAlgorithms[i]:
				standardMethod = i
				break
		data[''.join(map(str, stageSetup))] = listOutput

	resultPath = './results/data' + "".join(map(str, np.int0(defaultCost))) + '/'
	if not os.path.isdir(resultPath):
		os.mkdir(resultPath) 
	with open(resultPath +  'cov_' + str(alpha) + '_minCost.pickle', 'wb') as handle:
		pickle.dump(data, handle)
	# print('Done!')

def runAlgorithms(isQuickTest, df, metrics, algorithms, pi, mu = [ 0., 0., 0., 0.], 
			sigma = [	[ 	1.,		0.2,  	0.2, 	0.2   ], 
					[   0.2,  	1.,  	0.2,  	0.2   ],
					[   0.2,  	0.2,  	1.,  	0.2   ],
					[   0.2,  	0.2,  	0.2,  	1.   ]], 
			X0 = 10**5, c = [1,10,100,1000], lambdaLast = 0, alpha = 0.5, GT = 0, sampleEvaluation=False, stageSetup = 0, printOutput = False):

	numberOfStages = len(mu[0])
	violationTolerance  = 0.01

	dicOutput = {}
	dicOutput['algorithms'] = algorithms

	print("Number of true candidates: {:.0f}".format(GT))
	lambdas = [0]*(numberOfStages-1) + [lambdaLast]
	lambdas[:-1], dicOutput[algorithms[0] + '_cStar'], dicOutput[algorithms[0] + '_time'] = min_f_c(isQuickTest, pi, mu, sigma, lambdas[-1], c, alpha)
	_, samples = validateOperator(lambdas, pi, mu, sigma, c, X0, violationTolerance, algorithm=algorithms[0], printOutput = printOutput)
	dicOutput[algorithms[0] + '_cost'], dicOutput[algorithms[0] + '_samples'], dicOutput[algorithms[0] + 'sensitivity'], dicOutput[algorithms[0] + 'specificity'], dicOutput[algorithms[0] + 'F1'], dicOutput[algorithms[0] + 'accuracy'], dicOutput['sensitivity'], dicOutput['specificity'], dicOutput['F1'], dicOutput['accuracy'] = validateOperatorBySamplesNew(df.copy(), metrics, lambdas, samples, pi, mu, sigma, c, X0, violationTolerance, stageSetup, algorithm=algorithms[0], printOutput = printOutput)
	dicOutput[algorithms[0] + '_lambdas'] = lambdas
	dicOutput[algorithms[0] + '_totalCost'] = sum(dicOutput[algorithms[0] + '_cost'])

	lambdas = [0]*(numberOfStages-1) + [lambdaLast]
	lambdas[:-1], dicOutput[algorithms[1] + '_decayRatio'], dicOutput[algorithms[1] + '_time'], samples = baseApproachMine(df.copy(), metrics, dicOutput[algorithms[0] + '_samples'][-1], stageSetup, pi, mu, sigma, lambdas[-1], sum(dicOutput[algorithms[0] + '_cost']), c, X0, violationTolerance)
	dicOutput[algorithms[1] + '_cost'], dicOutput[algorithms[1] + '_samples'], dicOutput[algorithms[1] + 'sensitivity'], dicOutput[algorithms[1] + 'specificity'], dicOutput[algorithms[1] + 'F1'], dicOutput[algorithms[1] + 'accuracy'], _, _, _, _ = validateOperatorBySamplesNew(df.copy(), metrics, lambdas, samples, pi, mu, sigma, c, X0, violationTolerance, stageSetup, algorithm=algorithms[1], printOutput = printOutput)
	dicOutput[algorithms[1] + '_lambdas'] = lambdas
	dicOutput[algorithms[1] + '_totalCost'] = sum(dicOutput[algorithms[1] + '_cost'])
	return dicOutput

def initialFunction(x, pi, mu, sigma, C_total, c, X0, target):
	lowerBound = [-np.Inf]*(len(mu[0])-1) + [x[0]]
	AUC = (pi[0]*mvn.mvnun(lowerBound, [np.Inf]*len(mu[0]), mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, [np.Inf]*len(mu[1]), mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
	return (target- (AUC*X0))

def min_f_c(isQuickTest, pi, mu, sigma, lambdaLast, c, alpha):
	p = problem(pi, mu, sigma, lambdaLast, c, alpha)
	SD_0 = np.sqrt(np.diag(sigma[0]))
	lower_0 = mu[0][:-1] - (8*SD_0[:-1])
	upper_0 = mu[0][:-1] + (8*SD_0[:-1])
	SD_1 = np.sqrt(np.diag(sigma[1]))
	lower_1 = mu[1][:-1] - (8*SD_1[:-1])
	upper_1 = mu[1][:-1] + (8*SD_1[:-1])
	bounds = Bounds(np.min(np.vstack([lower_0, lower_1]), axis = 0), np.max(np.vstack([upper_0, upper_1]), axis = 0))

	start = time.time()
	if isQuickTest:
		result = differential_evolution(p.objectiveFunction, bounds)
	else:
		result = differential_evolution(p.objectiveFunction, bounds, tol=10**(-20), mutation=[0.05, 1], workers = 10)
	end = time.time()
	print(result.fun)

	return result.x, result.fun, (end - start)

class problem:
	def __init__(self, pi, mu, sigma, lambdaLast, c, alpha):
		# Score distribution
		self.pi = pi
		self.mu = mu
		self.sigma = sigma
		# Given threshold
		self.lambdaLast = lambdaLast
		# Simulation cost
		self.alpha = alpha
		self.c = c

	def objectiveFunction(self, lambdas):
		LB = [-np.Inf]*len(self.mu[0])
		f = self.c[0]
		for i in range(len(lambdas)):
			LB[i] = lambdas[i]
			AUC = (self.pi[0]*mvn.mvnun(LB, [np.Inf]*len(self.mu[0]), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(LB, [np.Inf]*len(self.mu[0]), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
			f += (self.c[i+1]*AUC)
		c1 = (self.pi[0]*mvn.mvnun(np.append([-np.Inf]*len(lambdas), self.lambdaLast), [np.Inf]*len(self.mu[0]), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) 
		c1 += (self.pi[1]*mvn.mvnun(np.append([-np.Inf]*len(lambdas), self.lambdaLast), [np.Inf]*len(self.mu[1]), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
		c2 = (self.pi[0]*mvn.mvnun(np.append(lambdas, self.lambdaLast), [np.Inf]*len(self.mu[0]), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) 
		c2 += (self.pi[1]*mvn.mvnun(np.append(lambdas, self.lambdaLast), [np.Inf]*len(self.mu[1]), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
		c = ((c1 - c2)/c1)
		return ( ( (1 - self.alpha) * ( f/(len(self.c)*self.c[-1]) ) ) + (self.alpha * c) )

def integrateCost(pi, mu, sigma, lowerBound, upperBound):
	return (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])

def baseApproachMine(dfOriginal, metrics, targetSamples, stageSetup, pi, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	start = time.time()
	# decayRatio = brentq(findDecayRatio, 0, 1, args=(targetSamples, dfOriginal, metrics, lambdaLast, c, X0))
	decayRatio = 0.75 # pass 30 % of samples
	lambdas = [-np.Inf]*(len(mu[0])-1)
	reorderedTable = dfOriginal[metrics]
	numberOfPassedSamples = [X0] + [0]*(len(metrics))
	for i in range(len(metrics)):
		if i < (len(metrics)-1):
			oneAddreorderedTable = reorderedTable.nlargest(int(decayRatio*len(reorderedTable)) + 1, metrics[i])
			reorderedTable = reorderedTable.nlargest(int(decayRatio*len(reorderedTable)), metrics[i])
			lambdas[i] = (np.min(reorderedTable[metrics[i]]) + np.min(oneAddreorderedTable[metrics[i]]))/2
			numberOfPassedSamples[i+1] = len(reorderedTable)
		else:
			reorderedTable = reorderedTable[reorderedTable[metrics[i]] > lambdaLast]
			numberOfPassedSamples[i+1] = len(reorderedTable)
	end = time.time()
	return lambdas, decayRatio, (end - start), numberOfPassedSamples

def findDecayRatio(r, targetSamples, df, metrics, lambdaLast, c, X0):
	reorderedTable = df[metrics]
	computationalCostVector = [0]*(len(metrics))
	numberOfPassedSamples = [X0] + [0]*(len(metrics))
	for i in range(len(metrics)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		
		if i < (len(metrics)-1):
			reorderedTable = reorderedTable.nlargest(int(r*len(reorderedTable)), metrics[i])
			numberOfPassedSamples[i+1] = len(reorderedTable)
		else:
			reorderedTable = reorderedTable[reorderedTable[metrics[i]] > lambdaLast]
			numberOfPassedSamples[i+1] = len(reorderedTable)

	return (targetSamples - numberOfPassedSamples[-1])

def baselineFindLambdas(x, i, LB, pi, mu, sigma, lambdaLast, c, X0, decayRatio):
	LB[i] = x[0]
	Xi = integrateCost(pi, mu, sigma, LB + [-np.Inf], [np.Inf]*len(mu[0]))
	return (Xi - (decayRatio**(i+1)))

def baseApproach2(pi, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	start = time.time()
	lambdas = [-np.Inf]*(len(mu[0])-1)
	costAvailable = C_total
	for i in range(len(lambdas)):
		isSolutionProper = False
		initialPoint = 0
		numberOfTrial = 0
		costAvailable -= (c[i] * X0 * integrateCost(pi, mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu[0])))
		costAvailableNextStage = costAvailable/ (len(mu[0])-(i+1))
		if costAvailableNextStage >= ( c[i+1]*X0*integrateCost(pi, mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu[0]))):
			lambdas[i] = -np.Inf
		else:
			while not isSolutionProper:
				lambdas[i] = fsolve(baselineFunction2, initialPoint, args=(i, lambdas, pi, mu, sigma, lambdaLast, C_total, c, X0, costAvailableNextStage), maxfev=1000000)[0]
				costExpected = c[i+1]*X0*integrateCost(pi, mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu[0]))
				if (np.abs(costAvailableNextStage - costExpected) < (costAvailableNextStage*0.001)) and ((costAvailableNextStage*1.001) >= costExpected):
					isSolutionProper = True
				else:
					numberOfTrial += 1
					if ((numberOfTrial%5000) == 0) and (numberOfTrial > 50):
						print("No solution was found retry: " + str(numberOfTrial))
					initialPoint = np.random.uniform(-20, 20, 1)
					if numberOfTrial > 10000:
						print("No solution was found retry: " + str(costExpected))
						break
	end = time.time()
	f_star = 1 - integrateCost(pi, mu, sigma, lambdas + [lambdaLast], [np.Inf]*len(mu[0]))
	return lambdas, f_star, (end - start)

def baselineFunction2(x, i, LB, pi, mu, sigma, lambdaLast, C_total, c, X0, availableCost):
	LB[i] = x[0]
	AUC = (pi[0] * mvn.mvnun(LB + [-np.Inf], [np.Inf]*len(mu[0]), mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1] * mvn.mvnun(LB + [-np.Inf], [np.Inf]*len(mu[1]), mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
	Xi = X0*AUC
	return (availableCost - (c[i+1]*Xi))

def validateOperator(x, pi, mu, sigma, c, X0, violationTolerance, algorithm = "unknown", printOutput = True):
	print(algorithm + ": {}".format(x))

	upperBound = [np.Inf]*len(mu[0])
	lowerBound = [-np.Inf]*len(mu[0])
	computationalCostVector = [0]*(len(mu[0]))
	numberOfPassedSamples = [X0] + [0]*(len(mu[0]))

	for i in range(len(x)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		lowerBound[i] = x[i]
		AUC = (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + ((pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0]))
		numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*AUC
		if printOutput:
			print("\tStage{}(cost:{:.0f}*samples:{:.2f}={:.2f}) - samples passed:{:.2f}".format(i+1, c[i], numberOfPassedSamples[i], c[i]* numberOfPassedSamples[i], numberOfPassedSamples[i+1]))
	
	AUC = (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
	f_star = (1 - AUC)
	print("\tSamples passed:{:.2f}, Total computational cost: {}".format(numberOfPassedSamples[-1], sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples

def validateOperatorBySamples(dfOriginal, metrics, x, samples, pi, mu, sigma, c, X0, violationTolerance, stageSetup, algorithm = "unknown", printOutput = True):
	dfPositive = dfOriginal.copy()

	computationalCostVector = [0]*(len(mu[0]))
	numberOfPassedSamples = [X0] + [0]*(len(mu[0]))
	for i in range(len(x)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		
		if i < (len(x)-1):
			dfPositive = dfPositive.nlargest(int(samples[i+1]), metrics[i])
			numberOfPassedSamples[i+1] = len(dfPositive)
		else:
			dfPositive = dfPositive[dfPositive[metrics[i]] > x[-1]]
			numberOfPassedSamples[i+1] = len(dfPositive)

		if printOutput:
			print("\tStage{}(cost:{:.0f}*samples:{:.2f}={:.2f}) - samples passed:{:.2f}".format(i+1, c[i], numberOfPassedSamples[i], c[i]* numberOfPassedSamples[i], numberOfPassedSamples[i+1]))

	dfNetagive = pd.concat([dfOriginal, dfPositive, dfPositive]).drop_duplicates(keep=False)
	TP = len(dfPositive[dfPositive['Label'] == 'ncRNA'])
	FP = len(dfPositive[dfPositive['Label'] != 'ncRNA'])
	TN = len(dfNetagive[dfNetagive['Label'] != "ncRNA"])
	FN = len(dfNetagive[dfNetagive['Label'] == "ncRNA"])
	sensitivity = TP/(TP + FN)
	specificity = TN/(TN + FP)
	F1 = (2*TP)/((2*TP) + FP + FN)
	accuracy = (TP + TN)/(TP + TN + FP + FN)

	dfPositiveRef = dfOriginal[dfOriginal[metrics[-1]] > x[-1]]
	dfNetagiveRef = pd.concat([dfOriginal, dfPositiveRef, dfPositiveRef]).drop_duplicates(keep=False)
	TPR = len(dfPositiveRef[dfPositiveRef['Label'] == 'ncRNA'])
	FPR = len(dfPositiveRef[dfPositiveRef['Label'] != 'ncRNA'])
	TNR = len(dfNetagiveRef[dfNetagiveRef['Label'] != "ncRNA"])
	FNR = len(dfNetagiveRef[dfNetagiveRef['Label'] == "ncRNA"])
	sensitivityRef = TPR/(TPR + FNR)
	specificityRef  = TNR/(TNR + FPR)
	F1Ref = (2*TPR)/((2*TPR) + FPR + FNR)
	accuracyRef = (TPR + TNR)/(TPR + TNR + FPR + FNR)

	print("\tSamples passed:{:.2f}, Total computational cost: {}".format(numberOfPassedSamples[-1], sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples, sensitivity, specificity, F1, accuracy, sensitivityRef, specificityRef, F1Ref, accuracyRef

def validateOperatorBySamplesNew(dfOriginal, metrics, x, samples, pi, mu, sigma, c, X0, violationTolerance, stageSetup, algorithm = "unknown", printOutput = True):
	print(algorithm + ": {}".format(x))
	dfPositive = dfOriginal.copy()

	computationalCostVector = [0]*(len(mu[0]))
	numberOfPassedSamples = [X0] + [0]*(len(mu[0]))
	for i in range(len(x)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		
		dfPositive = dfPositive[dfPositive[metrics[i]] > x[i]]
		numberOfPassedSamples[i+1] = len(dfPositive)

		if printOutput:
			print("\tStage{}(cost:{:.0f}*samples:{:.2f}={:.2f}) - samples passed:{:.2f}".format(i+1, c[i], numberOfPassedSamples[i], c[i]* numberOfPassedSamples[i], numberOfPassedSamples[i+1]))

	dfNetagive = pd.concat([dfOriginal, dfPositive, dfPositive]).drop_duplicates(keep=False)
	TP = len(dfPositive[dfPositive['Label'] == 'ncRNA'])
	FP = len(dfPositive[dfPositive['Label'] != 'ncRNA'])
	TN = len(dfNetagive[dfNetagive['Label'] != "ncRNA"])
	FN = len(dfNetagive[dfNetagive['Label'] == "ncRNA"])
	sensitivity = TP/(TP + FN)
	specificity = TN/(TN + FP)
	F1 = (2*TP)/((2*TP) + FP + FN)
	accuracy = (TP + TN)/(TP + TN + FP + FN)

	dfPositiveRef = dfOriginal[dfOriginal[metrics[-1]] > x[-1]]
	dfNetagiveRef = pd.concat([dfOriginal, dfPositiveRef, dfPositiveRef]).drop_duplicates(keep=False)
	TPR = len(dfPositiveRef[dfPositiveRef['Label'] == 'ncRNA'])
	FPR = len(dfPositiveRef[dfPositiveRef['Label'] != 'ncRNA'])
	TNR = len(dfNetagiveRef[dfNetagiveRef['Label'] != "ncRNA"])
	FNR = len(dfNetagiveRef[dfNetagiveRef['Label'] == "ncRNA"])
	sensitivityRef = TPR/(TPR + FNR)
	specificityRef  = TNR/(TNR + FPR)
	F1Ref = (2*TPR)/((2*TPR) + FPR + FNR)
	accuracyRef = (TPR + TNR)/(TPR + TNR + FPR + FNR)

	print("\tSamples passed:{:.2f}, Total computational cost: {}".format(numberOfPassedSamples[-1], sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples, sensitivity, specificity, F1, accuracy, sensitivityRef, specificityRef, F1Ref, accuracyRef

if __name__ == "__main__":
	# mainDefault()
	mainScreeningNcRNA()