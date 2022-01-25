import os
from re import A
import time
import numpy as np

from numpy.linalg import inv
from scipy import integrate
from scipy.optimize import differential_evolution, Bounds, fsolve, brentq
from scipy.stats import mvn, wishart, invwishart
from itertools import chain, combinations, permutations

from datetime import datetime
import pickle

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

def main():
	isQuickTest = True
	prinprintOutput = True
	algorithms = ['proposedLambda', 'baseline']
	# algorithms = ['baseline']

	evaluateJointOptimization(0, algorithms, "MLE", prinprintOutput, isQuickTest)
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
			[	[ 	1.,		0.5,  	0.4, 	0.3   ], 	# 9
				[   0.5,  	1.,  	0.5,  	0.4   ],
				[   0.4,  	0.5,  	1.,  	0.5   ],
				[   0.3,  	0.4,  	0.5,  	1.    ]	],	
			[	[ 	1.,		0.8,  	0.7, 	0.6   ], 	# 10
				[   0.8,  	1.,  	0.8,  	0.7   ],
				[   0.7,  	0.8,  	1.,  	0.8   ],
				[   0.6,  	0.7,  	0.8,  	1.    ]	]]

	# Experimental configuration
	X0 = 10**5

	return mu, sigma, X0

def evaluateJointOptimization(costDistSelector, algorithms, densityEstimation = "MLE", prinprintOutput = False, isQuickTest = False):
	alpha = 0.5
	mu, sigma, X0 = getDefaultParameters4()

	stageOrder = [0, 1, 2, 3]
	if costDistSelector == 0:
		defaultCost = [1, 10, 100, 1000]
	elif costDistSelector == 1:
		defaultCost = [1, 2, 4, 50]
	elif costDistSelector == 2:
		defaultCost = [1, 2, 4, 8]
	elif costDistSelector == 3:
		defaultCost = [4.9, 6.6, 5.7, 16.7]
	elif costDistSelector == 4:
		defaultCost = [0.001, 0.1, 10, 10000]

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
		if outerIndex < 10:
			continue
		data = {}
		i = 0
		for stageSetup in stageSetups: 
			if len(stageSetup) != 4:
				continue
			stageSetup = np.array(stageSetup)
			reorderedSigma = np.array(sigma[outerIndex])
			reorderedSigma = reorderedSigma[:, stageSetup][stageSetup]
			reorderedMu = np.array(mu)
			reorderedMu = reorderedMu[stageSetup]
			reorderedC = np.array(defaultCost)
			reorderedC = reorderedC[stageSetup]

			listOutput = list() 
			dicOutput = runAlgorithmsForJointOptimization(isQuickTest, algorithms, prinprintOutput, reorderedMu, reorderedSigma, X0, reorderedC, 0, alpha)
			listOutput.append(dicOutput)
			outAlgorithms = dicOutput['algorithms']
			standardMethod = 0
			for i in range(len(outAlgorithms)):
				if 'proposed' in outAlgorithms[i]:
					standardMethod = i
					break
			data[''.join(map(str, stageSetup))] = listOutput

		resultPath = './results/data' + "".join(map(str, defaultCost)) + '/'
		if not os.path.isdir(resultPath):
			os.mkdir(resultPath) 
		with open(resultPath + 'cov' + str(outerIndex) + '_' + str(alpha)+ '_minCost.pickle', 'wb') as handle:
			pickle.dump(data, handle)
		# print(str(outerIndex + 1) + "/" + str(len(sigma)) + ' Done!')

def runAlgorithmsForJointOptimization(isQuickTest, algorithms, printOutput = False, mu = [ 0., 0., 0., 0.], 
			sigma = [	[ 	1.,		0.2,  	0.2, 	0.2   ], 
					[   0.2,  	1.,  	0.2,  	0.2   ],
					[   0.2,  	0.2,  	1.,  	0.2   ],
					[   0.2,  	0.2,  	0.2,  	1.   ]], 
			X0 = 10**5, c = [1,10,100,1000], C_total = (100*2*(10**5)), alpha = 0.5):

	numberOfStages = len(mu)
	violationTolerance  = 0.01

	dicOutput = {}
	dicOutput['algorithms'] = algorithms

	lambdaLast = fsolve(initialFunction, 0, args = (mu, sigma, C_total, c, X0), maxfev=100000)[0]
	GT = X0*integrateCost(mu, sigma, [-np.Inf]*(len(mu)-1) + [lambdaLast], [np.Inf]*len(mu))
	print("Number of true candidates: {:.0f}".format(GT))

	lambdas = [0]*(numberOfStages-1) + [lambdaLast]
	lambdas[:-1], dicOutput[algorithms[0] + '_cStar'], dicOutput[algorithms[0] + '_time'] = min_f_c(isQuickTest, mu, sigma, lambdas[-1], c, alpha)
	dicOutput[algorithms[0] + '_cost'], dicOutput[algorithms[0] + '_samples'] = validateOperator(lambdas, mu, sigma, c, X0, violationTolerance, algorithm = algorithms[0], printOutput = printOutput)
	dicOutput[algorithms[0] + '_totalCost'] = sum(dicOutput[algorithms[0] + '_cost'])
	dicOutput[algorithms[0] + '_lambdas'] = lambdas

	lambdas = [0]*(numberOfStages-1) + [lambdaLast]
	lambdas[:-1], dicOutput[algorithms[1] + '_decayRatio'], dicOutput[algorithms[1] + '_time'] = baseApproachMine2(dicOutput[algorithms[0] + '_samples'][-1], mu, sigma, lambdas[-1], sum(dicOutput[algorithms[0] + '_cost']), c, X0, violationTolerance)
	dicOutput[algorithms[1] + '_cost'], dicOutput[algorithms[1] + '_samples'] = validateOperator(lambdas, mu, sigma, c, X0, violationTolerance, algorithm=algorithms[1], printOutput = printOutput)
	dicOutput[algorithms[1] + '_totalCost'] = sum(dicOutput[algorithms[1] + '_cost'])
	dicOutput[algorithms[1] + '_lambdas'] = lambdas
	return dicOutput

def min_f_c(isQuickTest, mu, sigma, lambdaLast, c, alpha):
	p = jointProblem(mu, sigma, lambdaLast, c, alpha)
	SD_0 = np.sqrt(np.diag(sigma))
	lower_0 = mu[:-1] - (8*SD_0[:-1])
	upper_0 = mu[:-1] + (8*SD_0[:-1])
	SD_1 = np.sqrt(np.diag(sigma))
	lower_1 = mu[:-1] - (8*SD_1[:-1])
	upper_1 = mu[:-1] + (8*SD_1[:-1])
	bounds = Bounds(np.min(np.vstack([lower_0, lower_1]), axis = 0), np.max(np.vstack([upper_0, upper_1]), axis = 0))

	start = time.time()
	if isQuickTest:
		result = differential_evolution(p.objectiveFunction, bounds, mutation=[0.1, 1])
	else:
		result = differential_evolution(p.objectiveFunction, bounds, tol=10**(-8), mutation=[0.05, 1])
	end = time.time()
	print(result.fun)

	return result.x, result.fun, (end - start)

class jointProblem:
	def __init__(self, mu, sigma, lambdaLast, c, alpha):
		# Score distribution
		self.mu = mu
		self.sigma = sigma
		# Given threshold
		self.lambdaLast = lambdaLast
		# Simulation cost
		self.alpha = alpha
		self.c = c

	def objectiveFunction(self, lambdas):
		LB = [-np.Inf]*len(self.mu)
		f = self.c[0]
		for i in range(len(lambdas)):
			LB[i] = lambdas[i]
			AUC = (mvn.mvnun(LB, [np.Inf]*len(self.mu), self.mu, self.sigma, maxpts=len(self.mu)*10000, abseps = 1e-10, releps = 1e-10)[0])
			f += (self.c[i+1]*AUC)
		c1 = (mvn.mvnun(np.append([-np.Inf]*len(lambdas), self.lambdaLast), [np.Inf]*len(self.mu), self.mu, self.sigma, maxpts=len(self.mu)*10000, abseps = 1e-40, releps = 1e-40)[0]) 
		c2 = (mvn.mvnun(np.append(lambdas, self.lambdaLast), [np.Inf]*len(self.mu), self.mu, self.sigma, maxpts=len(self.mu)*10000, abseps = 1e-40, releps = 1e-40)[0]) 
		c = ((c1 - c2)/c1)
		return ( ( (1 - self.alpha) * ( f/(len(self.c)*self.c[-1]) ) ) + (self.alpha * c) )

def initialFunction(x, mu, sigma, C_total, c, X0):
	lowerBound = [-np.Inf]*(len(mu)-1) + [x[0]]
	AUC = mvn.mvnun(lowerBound, [np.Inf]*len(mu), mu, sigma, maxpts=len(mu)*10000, abseps = 1e-10, releps = 1e-10)[0]
	return ((100) - (AUC*X0))

def integrateCost(mu, sigma, lowerBound, upperBound):
	return mvn.mvnun(lowerBound, upperBound, mu, sigma, maxpts=len(mu)*10000, abseps = 1e-10, releps = 1e-10)[0]

def baseApproachMine2(targetSamples, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	# version 3
	start = time.time()
	initialPoint = 2
	decayRatio = 0.1
	# decayRatio = 0.3 # pass 30 % of samples
	lambdas = [-np.Inf]*(len(mu)-1)
	for i in range(len(lambdas)):
		lambdas[i] = fsolve(baselineFindLambdas, initialPoint, args=(i, lambdas, mu, sigma, lambdaLast, c, X0, decayRatio), maxfev=1000000, factor = 10000, xtol=1.49012e-10)[0]
	end = time.time()
	f_star = 1 - integrateCost(mu, sigma, lambdas + [lambdaLast], [np.Inf]*len(mu))
	return lambdas, decayRatio, (end - start)

def baseApproachMine(targetSamples, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	# version 3
	start = time.time()
	isSolutionProper = False
	initialPoint = 0
	numberOfTrial = 0
	while not isSolutionProper:
		decayRatio = brentq(findDecayRatio, 0, 1, args=(targetSamples, mu, sigma, lambdaLast, c, X0))
		# decayRatio = 0.3 # pass 30 % of samples
		lambdas = [-np.Inf]*(len(mu)-1)
		for i in range(len(lambdas)):
			lambdas[i] = fsolve(baselineFindLambdas, initialPoint, args=(i, lambdas, mu, sigma, lambdaLast, c, X0, decayRatio), maxfev=1000000)[0]
		
		upperBound = [np.Inf]*len(mu)
		lowerBound = [-np.Inf]*len(mu)
		computationalCostVector = [0]*(len(mu))
		numberOfPassedSamples = [X0] + [0]*(len(mu))
		testLambdas = lambdas + [lambdaLast]
		for i in range(len(testLambdas)):
			computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
			lowerBound[i] = testLambdas[i]
			numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*integrateCost(mu, sigma, lowerBound, upperBound)
		computationalCostVector[-1] = c[-1]*numberOfPassedSamples[-1]
		if (np.abs(targetSamples - numberOfPassedSamples[-1]) <= (0.5)):
			isSolutionProper = True
		else:
			numberOfTrial += 1
			if ((numberOfTrial%10) == 0) and (numberOfTrial > 5):
				print("No solution was found retry: " + str(numberOfTrial))
			initialPoint = np.random.uniform(-20, 20, 1)
			if numberOfTrial > 200:
				print("(framework 2) No solution was found retry: " + str(sum(computationalCostVector)))
				break

	end = time.time()
	f_star = 1 - integrateCost(mu, sigma, lambdas + [lambdaLast], [np.Inf]*len(mu))
	return lambdas, decayRatio, (end - start)

def findDecayRatio(r, targetSamples, mu, sigma, lambdaLast, c, X0):
	lambdas = [-np.Inf]*(len(mu)-1)
	for i in range(len(lambdas)):
		initialPoint = 0
		# lambdas[i] = fsolve(baselineFindLambdas, initialPoint, args=(i, lambdas, mu, sigma, lambdaLast, c, X0, r[0]), maxfev=1000000)[0]
		lambdas[i] = fsolve(baselineFindLambdas, initialPoint, args=(i, lambdas, mu, sigma, lambdaLast, c, X0, r), maxfev=1000000)[0]
	numberOfSamples = X0*integrateCost(mu, sigma, lambdas + [lambdaLast], [np.Inf]*len(mu))
	return (targetSamples - numberOfSamples)

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

def validateOperator(x, mu, sigma, c, X0, violationTolerance, algorithm = "unknown", printOutput = True):
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
	print("\tSamples passed:{:.2f}, Total computational cost: {}".format(numberOfPassedSamples[-1], sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples

if __name__ == "__main__":
	main()