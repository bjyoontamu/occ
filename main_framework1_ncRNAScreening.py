import os
from re import A
import time
import numpy as np
import pandas as pd

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
	prinprintOutput = True
	algorithms = ['proposedLambda', 'baseline']

	evaluateOCC_ncRNAScreening(algorithms, "EM", prinprintOutput)
	print("Simulation END!!!!!")

def evaluateOCC_ncRNAScreening(algorithms, densityEstimation = "MLE", prinprintOutput = False):
	metrics = np.array(['CPC2', 'CPAT', 'PLEK', 'LncFinder'])

	df = pd.read_pickle("./data/v38._samples_score")
	df_n = df[df['Label'] != 'ncRNA']
	df_p = df[df['Label'] == 'ncRNA']	
	if densityEstimation == "MLE":
		# GMM - MLE
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
		# x_train = df[metrics].values
		x_train	= df[metrics].sample(frac = 0.04, random_state=1).values

		# fit a Gaussian Mixture Model with two components
		modelDistribution = GaussianMixture(n_components=2, covariance_type='full').fit(x_train)

		pi = modelDistribution.weights_
		mu_n = modelDistribution.means_[0]
		mu_p = modelDistribution.means_[1]
		sigma_n = modelDistribution.covariances_[0]
		sigma_p = modelDistribution.covariances_[1]

	# GENCODE
	defaultCost = np.array([0.002526541940798, 0.0027336052908625, 0.0831765095718637, 2.495623060544432])*1000

	X0 = len(df)
	Binitial = int((defaultCost[0] * X0)/2)
	Bend = defaultCost[-1] * X0
	Bstep = 10000000
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
		prevOperator = np.empty(len(stageSetup)-1)
		prevOperator[:] = np.nan

		evaluationRange1 = np.arange(0, defaultCost[-1]*X0, (defaultCost[-1]*X0)/10)
		evaluationRange = np.concatenate((evaluationRange1, np.array([defaultCost[-1]*X0])))
		# evaluationRange = np.arange(Binitial, Bend, Bstep)
		for CTotali in evaluationRange:
		# for CTotali in np.arange(Binitial, Bend, Bstep):
			dicOutput, prevOperator = runAlgorithms(df, metrics[stageSetup], algorithms, prevOperator, pi, reorderedMu, reorderedSigma, X0, reorderedC, CTotali, lambdaLast, GT, True, stageSetup, prinprintOutput)
			dicOutput['x'] = CTotali
			listOutput.append(dicOutput)
			outAlgorithms = dicOutput['algorithms']
			standardMethod = 0
			for i in range(len(outAlgorithms)):
				if 'proposed' in outAlgorithms[i]:
					standardMethod = i
					break
			if dicOutput[outAlgorithms[standardMethod] + '_samples'][-1] >= GT:
				break

		data[''.join(map(str, stageSetup))] = listOutput

	resultPath = './results/data' + "".join(map(str, np.int0(defaultCost))) + '/'
	if not os.path.isdir(resultPath):
		os.mkdir(resultPath) 
	with open(resultPath +  'cov.pickle', 'wb') as handle:
		pickle.dump(data, handle)
	print('Done!')

def runAlgorithms(df, metrics, algorithms, prevOperator, pi, mu = [ 0., 0., 0., 0.], 
			sigma = [	[ 	1.,		0.2,  	0.2, 	0.2   ], 
					[   0.2,  	1.,  	0.2,  	0.2   ],
					[   0.2,  	0.2,  	1.,  	0.2   ],
					[   0.2,  	0.2,  	0.2,  	1.   ]], 
			X0 = 10**5, c = [1,10,100,1000], C_total = (100*2*(10**5)), lambdaLast = 0, GT = 0, sampleEvaluation=False, stageSetup = 0, printOutput = False):

	numberOfStages = len(mu[0])
	violationTolerance  = 0.01

	dicOutput = {}
	dicOutput['algorithms'] = algorithms
	if (X0*c[0] >= C_total):
		print("!!!!Warning!!!! invalid parameter!")
		for i in range(len(algorithms)):
			dicOutput[algorithms[i] + '_samples'] = (0, 0)
			dicOutput[algorithms[i] + '_positiveSamples'] = 0
		return dicOutput, prevOperator

	print("Number of true candidates: {:.0f}".format(GT))
	for i in range(len(algorithms)):
		lambdas = [0]*(numberOfStages-1) + [lambdaLast]
		if algorithms[i] == 'proposedLambda':
			lambdas[:-1], dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'] = proposedApproach(pi, mu, sigma, lambdas[-1], C_total, c, X0, prevOperator)
			prevOperator = lambdas[:-1]
			_, samples = validateOperator(lambdas, pi, mu, sigma, C_total, c, X0, violationTolerance, algorithm=algorithms[i], printOutput = printOutput)
		elif algorithms[i] == 'baseline':
			# lambdas[:-1], dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'] = baseApproach2(pi, mu, sigma, lambdas[-1], C_total, c, X0, violationTolerance)
			lambdas[:-1], dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'], samples = baseApproachMine(df.copy(), metrics, dicOutput['proposedLambda_samples'][-1], stageSetup, pi, mu, sigma, lambdas[-1], C_total, c, X0, violationTolerance)
		if sampleEvaluation:
			dicOutput[algorithms[i] + '_cost'], dicOutput[algorithms[i] + '_samples'], dicOutput[algorithms[i] + '_positiveSamples']= validateOperatorBySamples(df.copy(), metrics, lambdas, samples, pi, mu, sigma, C_total, c, X0, violationTolerance, stageSetup, algorithm=algorithms[i], printOutput = printOutput)
		else:
			dicOutput[algorithms[i] + '_cost'], dicOutput[algorithms[i] + '_samples'] = validateOperator(lambdas, pi, mu, sigma, C_total, c, X0, violationTolerance, algorithm=algorithms[i], printOutput = printOutput)
		dicOutput[algorithms[i] + '_lambdas'] = lambdas
	return dicOutput, prevOperator

def initialFunction(x, pi, mu, sigma, C_total, c, X0, target):
	lowerBound = [-np.Inf]*(len(mu[0])-1) + [x[0]]
	AUC = (pi[0]*mvn.mvnun(lowerBound, [np.Inf]*len(mu[0]), mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, [np.Inf]*len(mu[1]), mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
	return (target- (AUC*X0))

def proposedApproach(pi, mu, sigma, lambdaLast, C_total, c, X0, prevOperator):
	p = problem(pi, mu, sigma, lambdaLast, C_total, c, X0)
	nlc = NonlinearConstraint(p.constraintFunction, -np.inf, C_total, hess = lambda x, v: np.zeros((len(mu[0])-1, len(mu[0])-1)))
	SD_0 = np.sqrt(np.diag(sigma[0]))
	lower_0 = mu[0][:-1] - (8*SD_0[:-1])
	upper_0 = mu[0][:-1] + (8*SD_0[:-1])
	SD_1 = np.sqrt(np.diag(sigma[1]))
	lower_1 = mu[1][:-1] - (8*SD_1[:-1])
	upper_1 = mu[1][:-1] + (8*SD_1[:-1])
	bounds = Bounds(np.min(np.vstack([lower_0, lower_1]), axis = 0), np.max(np.vstack([upper_0, upper_1]), axis = 0))
	C_used = 0
	iterationCount = 0
	bestLambdas = np.zeros(len(mu[0])-1)
	bestFunction = 0
	bestTime = 0
	bestBudget = 0
	while np.abs(C_total-C_used) > (C_total*0.1):
		if iterationCount > 0:
			print("It seems it is not good, lets try again ({})".format(iterationCount))
		start = time.time()
		# GENCODE
		if np.isnan(prevOperator).any():
			result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-20), mutation=[0.05, 1], workers = 10)
		else:
			result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-20), mutation=[0.05, 1], workers = 10, x0 = prevOperator)
		end = time.time()
		C_used =  p.findBudgetFromLambdas(np.append(result.x, lambdaLast))
		if np.abs(C_total-C_used) < np.abs(C_total-bestBudget):
			bestBudget = C_used
			bestLambdas = result.x
			bestTime = (end - start)
			bestFunction = result.fun
		iterationCount += 1
		if iterationCount > 100:
			print("OMG..")
			break
	return bestLambdas, bestFunction, bestTime

class problem:
	def __init__(self, pi, mu, sigma, lambdaLast, C_total, c, X0):
		# Score distribution
		self.pi = pi
		self.mu = mu
		self.sigma = sigma
		# Given threshold
		self.lambdaLast = lambdaLast
		# Simulation cost
		self.C_total = C_total
		self.c = c
		self.X0 = X0
		self.upperBound = [np.Inf]*len(mu[0])

	def objectiveFunction(self, lowerBound):
		op = (self.pi[0]*mvn.mvnun(np.append(lowerBound, self.lambdaLast), self.upperBound, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(np.append(lowerBound, self.lambdaLast), self.upperBound, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
		return (1 - op)

	def constraintFunction(self, lowerBound):
		LB = [-np.Inf]*len(self.mu[0])
		cumulativeCost = self.c[0]*self.X0
		for i in range(len(self.mu[0])-1):
			LB[i] = lowerBound[i]
			AUC = (self.pi[0]*mvn.mvnun(LB, self.upperBound, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(LB, self.upperBound, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
			cumulativeCost += (self.c[i+1]*self.X0*AUC)
		return np.array(cumulativeCost)

	def findBudgetFromLambdas(self, x):
		lowerBound = [-np.Inf]*len(self.mu[0])
		computationalCostVector = [0]*(len(self.mu[0]))
		numberOfPassedSamples = [self.X0] + [0]*(len(self.mu[0]))

		for i in range(len(x)):
			computationalCostVector[i] = (self.c[i]*numberOfPassedSamples[i])
			lowerBound[i] = x[i]
			AUC = (self.pi[0]*mvn.mvnun(lowerBound, self.upperBound, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(lowerBound, self.upperBound, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
			numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*AUC

		return np.sum(computationalCostVector)

def integrateCost(pi, mu, sigma, lowerBound, upperBound):
	return (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])

def baseApproachMine(dfOriginal, metrics, targetSamples, stageSetup, pi, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	start = time.time()
	decayRatio = brentq(findDecayRatio, 0, 1, args=(targetSamples, dfOriginal, metrics, pi, mu, sigma, lambdaLast, c, C_total, X0))
	lambdas = [-np.Inf]*(len(mu[0])-1)
	reorderedTable = dfOriginal[metrics]
	numberOfPassedSamples = [X0] + [0]*(len(metrics))
	totalComputationalCost = 0
	for i in range(len(metrics)):
		totalComputationalCost += (numberOfPassedSamples[i]*c[i])
		if i < (len(metrics)-1):
			oneAddreorderedTable = reorderedTable.nlargest(int(decayRatio*len(reorderedTable)) + 1, metrics[i])
			reorderedTable = reorderedTable.nlargest(int(decayRatio*len(reorderedTable)), metrics[i])
			lambdas[i] = (np.min(reorderedTable[metrics[i]]) + np.min(oneAddreorderedTable[metrics[i]]))/2
			numberOfPassedSamples[i+1] = len(reorderedTable)
		else:
			reorderedTable = reorderedTable[reorderedTable[metrics[i]] > lambdaLast]
			numberOfPassedSamples[i+1] = len(reorderedTable)
	end = time.time()
	if (C_total*1.0001 < totalComputationalCost) or (np.abs(C_total - totalComputationalCost) > (C_total*violationTolerance)):
		print("Baseline works incorrectly")

	return lambdas, decayRatio, (end - start), numberOfPassedSamples

def findDecayRatio(r, targetSamples, df, metrics, pi, mu, sigma, lambdaLast, c, C_total, X0):
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

	return (C_total - np.sum(computationalCostVector))

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

def validateOperator(x, pi, mu, sigma, C_total, c, X0, violationTolerance, algorithm = "unknown", printOutput = True):
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
	if C_total < sum(computationalCostVector):
		if np.abs(C_total - sum(computationalCostVector)) > (C_total*violationTolerance):
			print("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
	print("\tSamples passed:{:.2f}, Maximum cost allowed: {}, Total computational cost: {}".format(numberOfPassedSamples[-1], C_total, sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples

def validateOperatorBySamples(df, metrics, x, samples, pi, mu, sigma, C_total, c, X0, violationTolerance, stageSetup, algorithm = "unknown", printOutput = True):
	defaultCost = np.array([0.002526541940798, 0.0027336052908625, 0.0831765095718637, 2.495623060544432])*1000 # convert the cost in ms

	numberOfPositiveSamples = 0
	computationalCostVector = [0]*(len(mu[0]))
	numberOfPassedSamples = [X0] + [0]*(len(mu[0]))
	for i in range(len(x)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		
		if i < (len(x)-1):
			df = df.nlargest(int(samples[i+1]), metrics[i])
			numberOfPassedSamples[i+1] = len(df)
		else:
			df = df[df[metrics[i]] > x[-1]]
			numberOfPassedSamples[i+1] = len(df)
			numberOfPositiveSamples = len(df[df['Label'] == 'ncRNA'])
		if printOutput:
			print("\tStage{}(cost:{:.0f}*samples:{:.2f}={:.2f}) - samples passed:{:.2f}".format(i+1, c[i], numberOfPassedSamples[i], c[i]* numberOfPassedSamples[i], numberOfPassedSamples[i+1]))

	if C_total < sum(computationalCostVector):
		if np.abs(C_total - sum(computationalCostVector)) > (C_total*violationTolerance):
			print("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
	print("\tSamples passed:{:.2f}, Maximum cost allowed: {}, Total computational cost: {}".format(numberOfPassedSamples[-1], C_total, sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples, numberOfPositiveSamples

if __name__ == "__main__":
	mainScreeningNcRNA()