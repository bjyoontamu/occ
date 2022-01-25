import os
from re import A
import time
import numpy as np

from numpy.linalg import inv
from scipy import integrate
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds, fsolve, brentq
from scipy.stats import mvn, wishart, invwishart
from itertools import chain, combinations, permutations

from datetime import datetime
import pickle

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

def main():
	isQuickTest = False
	prinprintOutput = True
	algorithms = ['proposedLambda', 'baseline']

	evaluateAlgorithmsWithCertainty(0, algorithms, prinprintOutput, isQuickTest)
	if isQuickTest:
		print("Quick test done.")
	else:
		print("Simulation done.")

def getDefaultParameters4():
	# Model configuration
	mu = [ 0., 0., 0., 0.]

	sigma = [
			[	[ 	1.,		0.2,  	0.2, 	0.2   ],   	# 0
				[   0.2,  	1.,  	0.2,  	0.2   ],
				[   0.2,  	0.2,  	1.,  	0.2   ],
				[   0.2,  	0.2,  	0.2,  	1.    ] ],
			[	[ 	1.,		0.5,  	0.5, 	0.5   ],  	# 1
				[   0.5,  	1.,  	0.5,  	0.5   ],
				[   0.5,  	0.5,  	1.,  	0.5   ],
				[   0.5,  	0.5,  	0.5,  	1.    ] ],
			[	[ 	1.,		0.8,  	0.8, 	0.8   ], 	# 2
				[   0.8,  	1.,  	0.8,  	0.8   ],
				[   0.8,  	0.8,  	1.,  	0.8   ],
				[   0.8,  	0.8,  	0.8,  	1.    ]	],
			[	[ 	1.,		0.2,  	0.2, 	0.8   ], 	# 3
				[   0.2,  	1.,  	0.2,  	0.2   ],	
				[   0.2,  	0.2,  	1.,  	0.2   ],
				[   0.8,  	0.2,  	0.2,  	1.    ]	],
			[	[ 	1.,		0.2,  	0.2, 	0.2   ], 	# 4
				[   0.2,  	1.,  	0.2,  	0.8   ],
				[   0.2,  	0.2,  	1.,  	0.2   ],
				[   0.2,  	0.8,  	0.2,  	1.    ]	],
			[	[ 	1.,		0.2,  	0.2, 	0.2   ], 	# 5
				[   0.2,  	1.,  	0.2,  	0.2   ],
				[   0.2,  	0.2,  	1.,  	0.8   ],
				[   0.2,  	0.2,  	0.8,  	1.    ]	],	
			[	[ 	1.,		0.2,  	0.2, 	0.2   ], 	# 6
				[   0.2,  	1.,  	0.8,  	0.8   ],
				[   0.2,  	0.8,  	1.,  	0.8   ],
				[   0.2,  	0.8,  	0.8,  	1.    ]	],
			[	[ 	1.,		0.2,  	0.8, 	0.8   ], 	# 7
				[   0.2,  	1.,  	0.2,  	0.2   ],
				[   0.8,  	0.2,  	1.,  	0.8   ],
				[   0.8,  	0.2,  	0.8,  	1.    ]	],
			[	[ 	1.,		0.8,  	0.2, 	0.8   ], 	# 8
				[   0.8,  	1.,  	0.2,  	0.8   ],
				[   0.2,  	0.2,  	1.,  	0.2   ],
				[   0.8,  	0.8,  	0.2,  	1.    ]	],
			[	[ 	1.,		0.3,  	0.2, 	0.1   ], 	# 9
				[   0.3,  	1.,  	0.3,  	0.2   ],
				[   0.2,  	0.3,  	1.,  	0.3   ],
				[   0.1,  	0.2,  	0.3,  	1.    ]	],	
			[	[ 	1.,		0.8,  	0.7, 	0.6   ], 	# 10
				[   0.8,  	1.,  	0.8,  	0.7   ],
				[   0.7,  	0.8,  	1.,  	0.8   ],
				[   0.6,  	0.7,  	0.8,  	1.    ]	]]

	# Experimental configuration
	X0 = 10**5

	return mu, sigma, X0

def evaluateAlgorithmsWithCertainty(costDistSelector, algorithms, prinprintOutput = False, isQuickTest = False):
	mu, sigma, X0 = getDefaultParameters4()
	customRange = False
	stageOrder = [0, 1, 2, 3]
	if costDistSelector == 0:
		customRange = True
		if isQuickTest:
			defaultCost = [1, 10, 100, 1000]
			Binitial = int(X0/2)
			Bend = defaultCost[-1]*X0
			Bstep = 2000000
			evaluationRange = np.arange(Binitial, Bend, Bstep)
		else: 
			defaultCost = [1, 10, 100, 1000]
			Binitial = int(X0/defaultCost[0])
			Bend = defaultCost[-1]*X0
			evaluationRange0 = np.arange(0, int(X0/defaultCost[0]), int(X0/defaultCost[0]))
			evaluationRange1 = np.arange(int(X0/defaultCost[0]), 1000000, 1000000/10)
			evaluationRange2 = np.arange(1000000, 100000000, (100000000 - 1000000)/10)
			evaluationRange3 = np.arange(100000000, defaultCost[-1]*X0, (defaultCost[-1]*X0 - 100000000)/30)
			evaluationRange = np.concatenate((evaluationRange0, evaluationRange1, evaluationRange2, evaluationRange3, np.array([defaultCost[-1]*X0])))
	elif costDistSelector == 1:
		defaultCost = [1, 2, 4, 50]
		Binitial = int(X0/2)
		Bend = defaultCost[-1]*X0
		Bstep = 20000
	elif costDistSelector == 2:
		defaultCost = [1, 2, 4, 8]
		Binitial = int(X0/2)
		Bend = defaultCost[-1]*X0
		Bstep = 20000
	elif costDistSelector == 3:
		defaultCost = [4.9, 6.6, 5.7, 16.7]
		Binitial = int(X0/2)
		Bend = defaultCost[-1]*X0
		Bstep = 100000
	elif costDistSelector == 4:
		customRange = True
		if isQuickTest:
			defaultCost = [0.001, 0.1, 10, 10000]
			evaluationRange1 = np.arange(0, 1000000, 1000000/10)
			evaluationRange2 = np.arange(1000000, 100000000, (100000000 - 1000000)/10)
			evaluationRange3 = np.arange(100000000, defaultCost[-1]*X0, (defaultCost[-1]*X0 - 100000000)/30)
			evaluationRange = np.concatenate((evaluationRange1, evaluationRange2, evaluationRange3, np.array([defaultCost[-1]*X0])))
		else: 
			defaultCost = [0.001, 0.1, 10, 10000]
			evaluationRange1 = np.arange(0, 1000000, 1000000/10)
			evaluationRange2 = np.arange(1000000, 100000000, (100000000 - 1000000)/10)
			evaluationRange3 = np.arange(100000000, defaultCost[-1]*X0, (defaultCost[-1]*X0 - 100000000)/30)
			evaluationRange = np.concatenate((evaluationRange1, evaluationRange2, evaluationRange3, np.array([defaultCost[-1]*X0])))

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

	for outerIndex in range(len(sigma)):
		# test
		if outerIndex < 9:
			continue

		data = {}
		i = 0
		for stageSetup in stageSetups: 
			# # test
			# if len(stageSetup) != 4 or (stageSetup[0] != 0 or stageSetup[1] != 1):
			# 	continue
			
			stageSetup = np.array(stageSetup)
			reorderedSigma = np.array(sigma[outerIndex])
			reorderedSigma = reorderedSigma[:, stageSetup][stageSetup]
			reorderedMu = np.array(mu)
			reorderedMu = reorderedMu[stageSetup]
			reorderedC = np.array(defaultCost)
			reorderedC = reorderedC[stageSetup]

			listOutput = list() 
			prevOperator = np.empty(len(stageSetup)-1)
			prevOperator[:] = np.nan
			
			if not customRange:
				evaluationRange = np.arange(Binitial, Bend, Bstep)
			for CTotali in evaluationRange:
				dicOutput, prevOperator = runAlgorithms(isQuickTest, algorithms, prevOperator, prinprintOutput, reorderedMu, reorderedSigma, X0, reorderedC, CTotali)
				dicOutput['x'] = CTotali
				listOutput.append(dicOutput)
				outAlgorithms = dicOutput['algorithms']
				standardMethod = 0
				for i in range(len(outAlgorithms)):
					if 'proposed' in outAlgorithms[i]:
						standardMethod = i
						break
				# if dicOutput[outAlgorithms[standardMethod] + '_samples'][-1] > 98:
				if isQuickTest:
					if dicOutput[outAlgorithms[standardMethod] + '_samples'][-1] > 95:
						break
				else:
					if dicOutput[outAlgorithms[standardMethod] + '_samples'][-1] > 99.5:
						break
			data[''.join(map(str, stageSetup))] = listOutput

		resultPath = './results/data' + "".join(map(str, defaultCost)) + '/'
		if not os.path.isdir(resultPath):
			os.mkdir(resultPath) 
		with open(resultPath + 'cov' + str(outerIndex) + '.pickle', 'wb') as handle:
			pickle.dump(data, handle)
		print(str(outerIndex + 1) + "/" + str(len(sigma)) + ' Done!')

def runAlgorithms(isQuickTest, algorithms, prevOperator, printOutput = False, mu = [ 0., 0., 0., 0.], 
			sigma = [	[ 	1.,		0.2,  	0.2, 	0.2   ], 
					[   0.2,  	1.,  	0.2,  	0.2   ],
					[   0.2,  	0.2,  	1.,  	0.2   ],
					[   0.2,  	0.2,  	0.2,  	1.   ]], 
			X0 = 10**5, c = [1,10,100,1000], C_total = (100*2*(10**5))):

	numberOfStages = len(mu)
	violationTolerance  = 0.01

	dicOutput = {}
	dicOutput['algorithms'] = algorithms
	if (X0*c[0] >= C_total):
		print("!!!!Warning!!!! invalid parameter!")
		for i in range(len(algorithms)):
			dicOutput[algorithms[i] + '_samples'] = (0, 0)
		return dicOutput, prevOperator

	lambdaLast = fsolve(initialFunction, 0, args = (mu, sigma, C_total, c, X0), maxfev=100000)[0]
	GT = X0*integrateCost(mu, sigma, [-np.Inf]*(len(mu)-1) + [lambdaLast], [np.Inf]*len(mu))
	print("Number of true candidates: {:.0f}".format(GT))
	for i in range(len(algorithms)):
		lambdas = [0]*(numberOfStages-1) + [lambdaLast]
		if algorithms[i] == 'proposedLambda':
			lambdas[:-1], dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'] = proposedApproach(isQuickTest, mu, sigma, lambdas[-1], C_total, c, X0, prevOperator)
			prevOperator = lambdas[:-1]
		elif algorithms[i] == 'baseline':
			# lambdas[:-1], dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'] = baseApproach2(mu, sigma, lambdas[-1], C_total, c, X0, violationTolerance)
			lambdas[:-1], dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'] = baseApproachMine(dicOutput['proposedLambda_samples'][-1], mu, sigma, lambdas[-1], C_total, c, X0, violationTolerance)
		dicOutput[algorithms[i] + '_cost'], dicOutput[algorithms[i] + '_samples'] = validateOperator(lambdas, mu, sigma, C_total, c, X0, violationTolerance, algorithm=algorithms[i], printOutput = printOutput)
		dicOutput[algorithms[i] + '_lambdas'] = lambdas
	return dicOutput, prevOperator

def initialFunction(x, mu, sigma, C_total, c, X0):
	lowerBound = [-np.Inf]*(len(mu)-1) + [x[0]]
	AUC = mvn.mvnun(lowerBound, [np.Inf]*len(mu), mu, sigma, maxpts=len(mu)*10000, abseps = 1e-10, releps = 1e-10)[0]
	return ((100) - (AUC*X0))

def proposedApproach(isQuickTest, mu, sigma, lambdaLast, C_total, c, X0, prevOperator):
	p = problem(mu, sigma, lambdaLast, C_total, c, X0)
	nlc = NonlinearConstraint(p.constraintFunction, -np.inf, C_total, hess = lambda x, v: np.zeros((len(mu)-1, len(mu)-1)))
	SD = np.sqrt(np.diag(sigma))
	# bounds = Bounds(mu[:-1] - (3.5*SD[:-1]), mu[:-1] + (3.5*SD[:-1]))
	bounds = Bounds(mu[:-1] - (8*SD[:-1]), mu[:-1] + (8*SD[:-1]))
	C_used = 0
	iterationCount = 0
	bestLambdas = np.zeros(len(mu)-1)
	bestFunction = 0
	bestTime = 0
	bestBudget = 0
	while np.abs(C_total-C_used) > (C_total*0.01):
		if iterationCount > 0:
			print("It seems it is not good, lets try again ({})".format(iterationCount))
			# print("It seems it is not good, lets try again ({})".format(iterationCount))
		start = time.time()
		if isQuickTest:
			if np.isnan(prevOperator).any():
				result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-8), mutation=[0.1, 1])
			else:
				result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-8), mutation=[0.1, 1], x0 = prevOperator)
		else:
			if np.isnan(prevOperator).any():
				result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-8), mutation=[0.05, 1])
			else:
				result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-8), mutation=[0.05, 1], x0 = prevOperator)

		# result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-5), mutation=[0.1, 1])
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
			print("OMG..")
			break
	return bestLambdas, bestFunction, bestTime

class problem:
	def __init__(self, mu, sigma, lambdaLast, C_total, c, X0):
		# Score distribution
		self.mu = mu
		self.sigma = sigma
		# Given threshold
		self.lambdaLast = lambdaLast
		# Simulation cost
		self.C_total = C_total
		self.c = c
		self.X0 = X0
		self.upperBound = [np.Inf]*len(mu)

	def objectiveFunction(self, lowerBound):
		return (1 - (mvn.mvnun(np.append(lowerBound, self.lambdaLast), self.upperBound, self.mu, self.sigma, maxpts=len(self.mu)*10000, abseps = 1e-10, releps = 1e-10)[0]))

	def constraintFunction(self, lowerBound):
		LB = [-np.Inf]*len(self.mu)
		cumulativeCost = self.c[0]*self.X0
		for i in range(len(self.mu)-1):
			LB[i] = lowerBound[i]
			cumulativeCost += (self.c[i+1]*self.X0*mvn.mvnun(LB, self.upperBound, self.mu, self.sigma, maxpts=len(self.mu)*10000, abseps = 1e-10, releps = 1e-10)[0])
		return np.array(cumulativeCost)

	def findBudgetFromLambdas(self, x):
		lowerBound = [-np.Inf]*len(self.mu)
		computationalCostVector = [0]*(len(self.mu))
		numberOfPassedSamples = [self.X0] + [0]*(len(self.mu))

		for i in range(len(x)):
			computationalCostVector[i] = (self.c[i]*numberOfPassedSamples[i])
			lowerBound[i] = x[i]
			numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*mvn.mvnun(lowerBound, self.upperBound, self.mu, self.sigma, maxpts=len(self.mu)*10000, abseps = 1e-10, releps = 1e-10)[0]

		return np.sum(computationalCostVector)

def integrateCost(mu, sigma, lowerBound, upperBound):
	return mvn.mvnun(lowerBound, upperBound, mu, sigma, maxpts=len(mu)*10000, abseps = 1e-10, releps = 1e-10)[0]

def baseApproachMine(targetSamples, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	# version 3
	start = time.time()
	isSolutionProper = False
	initialPoint = 0
	numberOfTrial = 0
	while not isSolutionProper:
		decayRatio = brentq(findDecayRatio, 0, 1, args=(targetSamples, mu, sigma, lambdaLast, c, C_total, X0), xtol=2e-30, rtol=8.881784197001252e-16, maxiter=10000)
		# decayRatio = 0.3 # pass 30 % of samples
		lambdas = [-np.Inf]*(len(mu)-1)
		for i in range(len(lambdas)):
			lambdas[i] = fsolve(baselineFindLambdas, initialPoint, args=(i, lambdas, mu, sigma, lambdaLast, c, X0, decayRatio), maxfev=1000000)[0]
		
		upperBound = [np.Inf]*len(mu)
		lowerBound = [-np.Inf]*len(mu)
		computationalCostVector = [0]*(len(mu))
		numberOfPassedSamples = [X0] + [0]*(len(mu) - 1)

		for i in range(len(lambdas)):
			computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
			lowerBound[i] = lambdas[i]
			numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*integrateCost(mu, sigma, lowerBound, upperBound)
		computationalCostVector[-1] = c[-1]*numberOfPassedSamples[-1]
		if (C_total*1.0001 >= sum(computationalCostVector)) and (np.abs(C_total - sum(computationalCostVector)) <= (C_total*violationTolerance)):
			isSolutionProper = True
		else:
			numberOfTrial += 1
			if ((numberOfTrial%10) == 0) and (numberOfTrial > 5):
				print("No solution was found retry: " + str(numberOfTrial))
			initialPoint = np.random.uniform(-20, 20, 1)
			if numberOfTrial > 500:
				print("No solution was found retry: " + str(sum(computationalCostVector)))
				break

	end = time.time()
	f_star = 1 - integrateCost(mu, sigma, lambdas + [lambdaLast], [np.Inf]*len(mu))
	return lambdas, decayRatio, (end - start)

def findDecayRatio(r, targetSamples, mu, sigma, lambdaLast, c, C_total, X0):
	costAvaiable = C_total
	for i in range(len(mu)):
		costAvaiable -= (c[i]*X0*(r**(i)))
	return costAvaiable

def baselineFindLambdas(x, i, LB, mu, sigma, lambdaLast, c, X0, decayRatio):
	LB[i] = x[0]
	Xi = integrateCost(mu, sigma, LB + [-np.Inf], [np.Inf]*len(mu))
	return (Xi - (decayRatio**(i+1)))

def baseApproach2(mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	start = time.time()
	lambdas = [-np.Inf]*(len(mu)-1)
	costAvailable = C_total
	for i in range(len(lambdas)):
		isSolutionProper = False
		initialPoint = 0
		numberOfTrial = 0
		costAvailable -= (c[i] * X0 * integrateCost(mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu)))
		costAvailableNextStage = costAvailable/ (len(mu)-(i+1))
		if costAvailableNextStage >= ( c[i+1]*X0*integrateCost( mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu) ) ):
			lambdas[i] = -np.Inf
		else:
			while not isSolutionProper:
				lambdas[i] = fsolve(baselineFunction2, initialPoint, args=(i, lambdas, mu, sigma, lambdaLast, C_total, c, X0, costAvailableNextStage), maxfev=1000000)[0]
				costExpected = c[i+1]*X0*integrateCost(mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu))
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
	f_star = 1 - integrateCost(mu, sigma, lambdas + [lambdaLast], [np.Inf]*len(mu))
	return lambdas, f_star, (end - start)

def baselineFunction2(x, i, LB, mu, sigma, lambdaLast, C_total, c, X0, availableCost):
	LB[i] = x[0]
	Xi = X0*mvn.mvnun(LB + [-np.Inf], [np.Inf]*len(mu), mu, sigma, maxpts=len(mu)*10000, abseps = 1e-10, releps = 1e-10)[0]
	return (availableCost - (c[i+1]*Xi))

def validateOperator(x, mu, sigma, C_total, c, X0, violationTolerance, algorithm = "unknown", printOutput = True):
	print(algorithm + ": {}".format(x))
		
	upperBound = [np.Inf]*len(mu)
	lowerBound = [-np.Inf]*len(mu)
	computationalCostVector = [0]*(len(mu))
	numberOfPassedSamples = [X0] + [0]*(len(mu))

	for i in range(len(x)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		lowerBound[i] = x[i]
		numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*mvn.mvnun(lowerBound, upperBound, mu, sigma, maxpts=len(mu)*10000, abseps = 1e-10, releps = 1e-10)[0]
		if printOutput:
			print("\tStage{}(cost:{:.0f}*samples:{:.2f}={:.2f}) - samples passed:{:.2f}".format(i+1, c[i], numberOfPassedSamples[i], c[i]* numberOfPassedSamples[i], numberOfPassedSamples[i+1]))
		
	f_star = (1 - mvn.mvnun(lowerBound, upperBound, mu, sigma, maxpts=len(mu)*10000, abseps = 1e-10, releps = 1e-10)[0])
	if C_total < sum(computationalCostVector):
		if np.abs(C_total - sum(computationalCostVector)) > (C_total*violationTolerance):
			print("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
			print("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
	print("\tSamples passed:{:.2f}, Maximum cost allowed: {}, Total computational cost: {}".format(numberOfPassedSamples[-1], C_total, sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples

if __name__ == "__main__":
	main()