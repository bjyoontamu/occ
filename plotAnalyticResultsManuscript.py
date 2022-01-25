import pickle
import os

import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from itertools import chain, combinations, permutations
import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='Times New Roman', size = 'x-small', style = 'italic')
plt.rcParams['mathtext.fontset'] = "stix"
plt.rcParams["font.family"] = "Times New Roman"

def main():
    restultPath = './results/data1101001000_manuscript/'
    # restultPath = './results/data1101001000/'
    # restultPath = './results/data0.0010.11010000/'
    # restultPath = './results/data1248/'
    # restultPath = './results/data46516/'
    # restultPath = './results/data4.96.65.716.7/'
    # outPath = './figures/'
    outPath = restultPath + 'figures/'

    maxLengthOfStages = 4
    stages = getStages(maxLengthOfStages, "all")
    bestPerformers = drawFinalSamplesPerBudget(stages, restultPath, restultPath + 'finalSamplesPerBudget/')

    metrics = ['lambdas', 'cost', 'samples']
    # metrics = ['lambdas']
    for metric in metrics:
        # drawMetricPerBudget(metric, [(0, 3), (1, 0, 3), (0, 1, 3), (0, 1, 2, 3), (1, 0, 2, 3)], restultPath, restultPath + metric + '/', drawAll = True)   
        drawMetricPerBudget(metric, bestPerformers, restultPath, restultPath + metric + '/', drawAll = False)   

def getStages(maxLengthOfStages, mode):
    stageOrder = list(range(maxLengthOfStages))
    allStageSetups = list(chain.from_iterable(combinations(stageOrder, r) for r in range(len(stageOrder)+1)))
    stageSetups = list()
    for i in range(len(allStageSetups)):
        stageSetup = allStageSetups[i]
        if (len(stageSetup) > 1) and (stageSetup[-1] == stageOrder[-1]):
            if mode == "all":
                # All combinations and permutations    
                if len(stageSetup) > 2:
                    allPermutations = list(permutations(stageSetup))
                    for permutation in allPermutations:
                        if (permutation[-1] == stageOrder[-1]):
                            stageSetups.append(permutation)
                else:
                    stageSetups.append(stageSetup)
                
                # # All combinations w.o. permutations  
                # stageSetups.append(stageSetup)

            elif mode == "4":
                # All permutations with four stages
                if len(stageSetup) == 4:
                    # stageSetups.append(stageSetup)
                    allPermutations = list(permutations(stageSetup))
                    for permutation in allPermutations:
                        if (permutation[-1] == stageOrder[-1]):
                            stageSetups.append(permutation)

    # # Select setups to draw
    # stageSetupsTemp = x=[[] for i in range(maxLengthOfStages)]
    # for stageSetup in stageSetups:
    #     stageSetupsTemp[len(stageSetup)-1].append(stageSetup)
    # stageSetupsToDraw = list()
    # for i in range(1, maxLengthOfStages):
    #     threshold = list()
    #     for stageSetup in stageSetupsTemp[i]:
    #         threshold.append(np.min(np.where(np.array(data["y_" + 'proposed' + '_' + ''.join(map(str, stageSetup))])>90)))
    #     for stageSetup in stageSetupsTemp[i]:
    #         if np.sort(threshold)[2] >= np.min(np.where(np.array(data["y_" + 'proposed' + '_' + ''.join(map(str, stageSetup))])>90)):
    #             stageSetupsToDraw.append(stageSetup)
    # Select all
    stageSetupsToDraw = stageSetups
    return stageSetupsToDraw

def drawFinalSamplesPerBudget(stages, restultPath, outPath):
    fontP = FontProperties()
    fontP.set_size('xx-small')

    if not os.path.isdir(outPath):
        os.mkdir(outPath)
    if not os.path.isdir(outPath + 'all'):
        os.mkdir(outPath + 'all')

    fileList = list()
    for root, dirs, files in os.walk(restultPath, topdown=True):
        for name in files:
            if not "minCost" in name:
                if not "DS_Store" in name:
                    fileList.append(root + name)
        break
    
    bestSetups2Return = {}
    for fileName in fileList:
        with open(fileName, 'rb') as handle:
            data = pickle.load(handle)

        # # if fileName != './results/data1101001000/cov9.pickle':
        # if fileName != './results/data1101001000/cov10.pickle':
        #     continue
        # # if fileName.replace(".pickle", "_0.5_minCost.pickle") != './results/data1101001000/cov10_0.5_minCost.pickle':
        # #     continue
        # with open(fileName.replace(".pickle", "_0.5_minCost.pickle"), 'rb') as handle:
        #     data_minCost = pickle.load(handle)

        maxDomain = []
        for stageSetup in stages:
            if len(data[''.join(map(str, stageSetup))]) > len(maxDomain):
                maxDomain = [d['x'] for d in data[''.join(map(str, stageSetup))]]

        dfList = list()
        algorithmList = []
        lStyle = {}
        
        # draw all
        # alternativeVal = int(len(maxDomain)/25)
        alternativeVal = 0
        if alternativeVal == 0: alternativeVal = 1
        for stageSetup in stages:
            for algorithm in data[''.join(map(str, stageSetup))][0]['algorithms']:
                if algorithm not in algorithmList:
                    algorithmList.append(algorithm)
                    dfList.append(pd.DataFrame(index=maxDomain[::alternativeVal]))
                column2Add = [d[algorithm + '_samples'][-1] for d in data[''.join(map(str, stageSetup))]] + ([NaN]*(len(maxDomain)-len([d[algorithm + '_samples'][-1] for d in data[''.join(map(str, stageSetup))]])))
                dfList[algorithmList.index(algorithm)]['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + ']$'] = column2Add[::alternativeVal]
                lStyle['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + ']$'] = getLineColor(stageSetup)
        
        dfList[0].to_excel("output.xlsx")
        for i in range(len(algorithmList)):
            dfList[i].plot(figsize=(7, 6), style=lStyle)
            # plt.axvline(x = data['costPerStage'][-1]*data['numberOfSamples'])
            # plt.hlines(y=data['GT'][-1], xmin=0.0, xmax=1.0, color='b')
            # plt.axhline(y = data['GT'])
            plt.title(algorithmList[i])
            # plt.xlabel('Total computational budget $C$')
            # plt.ylabel('The number of potential samples $\mathbb{Y}$')
            plt.xlabel('Total resource budget')
            plt.ylabel('The number of potential candidates')
            # plt.ylim(0, 20)
            # plt.ylim(0, 80)
            # plt.xlim(np.min(dfList[0].index)-1000, np.max(dfList[0].index)+1000)
            # plt.xlim(16000, 27000)
            plt.grid(True)
            plt.yticks(np.arange(0, 100, step=10))
            plt.savefig(outPath + 'all/' + fileName.replace(restultPath, "").replace(".pickle", "") + algorithmList[i] + ".pdf")
            # # maxIndex = 0
            # # for j in range(len(dfList[i].index)):
            # #     if np.max(dfList[i].iloc[j,:]) > 90:
            # #         maxIndex = j + 4
            # #         break
            # # if maxIndex < (len(dfList[i].index)/2):
            # #     plt.xlim(0, dfList[i].index[maxIndex])
            # #     plt.savefig(outPath + fileName.replace(restultPath, "").replace(".pickle", "") + algorithmList[i] + "_closeup.jpg")
            # plt.xlim(0, dfList[i].index[int((len(dfList[i].index)/3))])
            # plt.savefig(outPath + fileName.replace(restultPath, "").replace(".pickle", "") + algorithmList[i] + "_closeup.jpg")
            plt.close()

        for i in range(3):
            if i == 0 or i == 3:
                list2Plot = ['$[S_1,S_4]$', '$[S_2,S_4]$', '$[S_3,S_4]$']
                list2GrayPlot = ['$[S_1,S_2,S_4]$', '$[S_1,S_3,S_4]$', '$[S_2,S_1,S_4]$', '$[S_2,S_3,S_4]$', '$[S_3,S_1,S_4]$', '$[S_3,S_2,S_4]$', '$[S_1,S_2,S_3,S_4]$', '$[S_1,S_3,S_2,S_4]$', '$[S_2,S_1,S_3,S_4]$', '$[S_2,S_3,S_1,S_4]$', '$[S_3,S_1,S_2,S_4]$', '$[S_3,S_2,S_1,S_4]$']
            elif i == 1 or i == 4:
                list2Plot = ['$[S_1,S_2,S_4]$', '$[S_1,S_3,S_4]$', '$[S_2,S_1,S_4]$', '$[S_2,S_3,S_4]$', '$[S_3,S_1,S_4]$', '$[S_3,S_2,S_4]$']
                list2GrayPlot = ['$[S_1,S_4]$', '$[S_2,S_4]$', '$[S_3,S_4]$', '$[S_1,S_2,S_3,S_4]$', '$[S_1,S_3,S_2,S_4]$', '$[S_2,S_1,S_3,S_4]$', '$[S_2,S_3,S_1,S_4]$', '$[S_3,S_1,S_2,S_4]$', '$[S_3,S_2,S_1,S_4]$']
            elif i == 2 or i == 5:
                list2Plot = ['$[S_1,S_2,S_3,S_4]$', '$[S_1,S_3,S_2,S_4]$', '$[S_2,S_1,S_3,S_4]$', '$[S_2,S_3,S_1,S_4]$', '$[S_3,S_1,S_2,S_4]$', '$[S_3,S_2,S_1,S_4]$']
                list2GrayPlot = ['$[S_1,S_4]$', '$[S_2,S_4]$', '$[S_3,S_4]$', '$[S_1,S_2,S_4]$', '$[S_1,S_3,S_4]$', '$[S_2,S_1,S_4]$', '$[S_2,S_3,S_4]$', '$[S_3,S_1,S_4]$', '$[S_3,S_2,S_4]$'] 
            
            ax = plt.subplot(3, 1, i+1)
            dfList[0][list2Plot].plot(figsize=(7.5, 6.5), style = lStyle, linewidth=0.8, alpha=0.7, label=list2Plot, ax=ax)
            for v in list2GrayPlot:
                plt.plot(dfList[0].index, dfList[0][v], color='grey', linewidth=0.3, alpha=0.2)

            # dfList[1][list2Plot].plot(figsize=(7.5, 6.5), linestyle = '--', style = lStyle, linewidth=0.8, alpha=0.7, label=list2Plot, ax=ax)
            # for v in list2GrayPlot:
            #     plt.plot(dfList[1].index, dfList[0][v], color='grey', linewidth=0.3, alpha=0.2)
            # legend = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon = 1, fontsize = 'small', prop=font, ncol=2)
            legend = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon = 1, fontsize = 'small', prop=font)
            frame = legend.get_frame()
            frame.set_color('white')
            # plt.legend(bbox_to_anchor=(1, 1), fontsize = 'small', prop=font, loc=5, ncol=2)
            plt.axvline(x = 1000*(10**5), ls = '--', color = 'k', linewidth = 0.8)
            plt.axhline(y = 100, ls = '--', color = 'k', linewidth = 0.8)
            # Same limits for every chart
            plt.xlim(0, 1.2*1000*(10**5))
            # plt.ylim(-100, 55000)
            # if i == 2:
            #     plt.xlabel('Total computational budget $C$')
            # if i == 1:
            #     plt.ylabel('The number of potential candidates $\mathbb{Y}$')
            if i == 2:
                plt.xlabel('Total resource budget')
            if i == 1:
                plt.ylabel('The number of potential candidates')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(outPath + fileName.replace(restultPath, "").replace(".pickle", "") + "_Y.pdf", transparent=True)
        plt.close()


        alternativeVal = 0
        if alternativeVal == 0: alternativeVal = 1
        for stageSetup in stages:
            for algorithm in data[''.join(map(str, stageSetup))][0]['algorithms']:
                if algorithm not in algorithmList:
                    algorithmList.append(algorithm)
                    dfList.append(pd.DataFrame(index=maxDomain[::alternativeVal]))
                column2Add = [d[algorithm + '_samples'][-1] for d in data[''.join(map(str, stageSetup))]] + ([NaN]*(len(maxDomain)-len([d[algorithm + '_samples'][-1] for d in data[''.join(map(str, stageSetup))]])))
                dfList[algorithmList.index(algorithm)]['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + ']$'] = column2Add[::alternativeVal]
                lStyle['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + ']$'] = getLineColor2(stageSetup)
                       
        targetAlgorithm = 'proposedLambda'
        df = dfList[algorithmList.index(targetAlgorithm)]
        bestSetups = list()
        bestSetupsIndices = list()
        for i in range(len(df)):
            if np.max(df.iloc[i, :]) < 80:
                continue
            # if np.max(df.iloc[i, :]) > 90:
            #     continue
            if np.isnan(df.iloc[i, :]).any():
                break
            bestSetups += list(df.columns[np.where(df.iloc[i, :] >= (np.max(df.iloc[i, :])))])
            for bestSetupIndex in list(np.where(df.iloc[i, :].astype(int) >= (int(np.max(df.iloc[i, :]))))[0]):
                bestSetupsIndices.append(bestSetupIndex)
        bestSetups = list(set(bestSetups))
        bestSetups.sort() # sorts normally by alphabetical order
        bestSetups.sort(key=len) # sorts by descending length
        # # For manuscript
        # bestSetups = ['$[S_0,S_1,S_2,S_3]$', '$[S_0,S_1,S_3]$', '$[S_0,S_2,S_3]$']
        bestSetups = ['$[S_1,S_2,S_3,S_4]$', '$[S_2,S_3,S_4]$', '$[S_1,S_4]$', '$[S_2,S_4]$', '$[S_3,S_4]$']
        # bestSetups = ['$[S_1,S_2,S_3,S_4]$', '$[S_2,S_3,S_4]$']
        bestDf = df[bestSetups]
        bestDf = bestDf.add_prefix('proposed_')

        df = dfList[algorithmList.index('baseline')]
        baselineDf = df[bestSetups]
        baselineDf = baselineDf.add_prefix('baseline_')
        bestLineColor = {}
        bestLineStyle = {}
        for setup in bestSetups:
            bestLineColor['proposed_' + setup] = lStyle[setup]
            bestLineColor['baseline_' + setup] = lStyle[setup]
            # bestLineStyle['proposed_' + setup] = 'solid'
            # bestLineStyle['baseline_' + setup] = 'dashed'
        fig, ax = plt.subplots()
        bestDf.plot(figsize=(9, 2), style =  bestLineColor, linewidth = 1.5, alpha=0.85, ax=ax)
        # bestDf.plot(figsize=(4, 3.5), style =  bestLineColor, linewidth = 1, alpha=0.7, ax=ax)
        # baselineDf.plot(figsize=(4, 3.5), style =  bestLineColor, linewidth = 0.8, linestyle = 'dashed', alpha=0.5, ax=ax)
        # plt.xlabel('Total computational budget $C$')
        # plt.ylabel('The number of potential samples $\left|\mathbb{Y}\\right|$')
        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.8))
        plt.title('A. $\\rho = 0.8$ B. $\\rho = 0.3$')
        plt.xlabel('Total resource budget')
        plt.ylabel('The number of potential candidates')
        plt.xlim(-5000000, 5000000+10**8 )
        plt.ylim(0, 105)
        plt.grid(True)
        plt.axvline(x = 1000*(10**5), ls = '--', color = 'k', linewidth = 0.8)
        plt.axhline(y = 100, ls = '--', color = 'k', linewidth = 0.8)
        # np.concatenate((np.array([-.05*10**8]), np.arange(0, 1.01*10**8, step=10**7), np.array([1.05*10**8])))
        plt.xticks(np.arange(0, 1.01*10**8, step=10**7))
        plt.yticks(np.arange(0, 110, step=20))
        plt.savefig(outPath + 'all/' + fileName.replace(restultPath, "").replace(".pickle", "") + targetAlgorithm + "_best.pdf", bbox_inches='tight')
        plt.close()
        bestSetupsIndicesForCov = list()
        bestSetupsIndices = list(set(bestSetupsIndices))
        bestSetupsIndices.sort()
        for i in bestSetupsIndices:
            bestSetupsIndicesForCov.append(stages[i])
        bestSetups2Return[fileName] = bestSetupsIndicesForCov
    return bestSetups2Return

def getLineColor(stageSetup):
    lineOption = ''
    if len(stageSetup) == 2:
        # pastel
        if "".join(map(str,stageSetup)) == '03':
            lineOption += '#E13102'
        elif "".join(map(str,stageSetup)) == '13':
            lineOption += '#99E472'
        elif "".join(map(str,stageSetup)) == '23':
            lineOption += '#1DD5EE'

    if len(stageSetup) == 3:
        # First strong
        if "".join(map(str,stageSetup)) == '013':
            lineOption += '#B73A3A'
        elif "".join(map(str,stageSetup)) == '023':
            lineOption += '#F28A90'
        elif "".join(map(str,stageSetup)) == '103':
            lineOption += '#326633'
        elif "".join(map(str,stageSetup)) == '123':
            lineOption += '#57B956'
        elif "".join(map(str,stageSetup)) == '203':
            lineOption += '#094782'
        elif "".join(map(str,stageSetup)) == '213':
            lineOption += '#098BF5'
    
    if len(stageSetup) == 4:
        if "".join(map(str,stageSetup)) == '0123':
            lineOption += '#EC5656'
        elif "".join(map(str,stageSetup)) == '0213':
            lineOption += '#F8BCBD'
        elif "".join(map(str,stageSetup)) == '1023':
            lineOption += '#478F48'
        elif "".join(map(str,stageSetup)) == '1203':
            lineOption += '#71C0A7'
        elif "".join(map(str,stageSetup)) == '2013':
            lineOption += '#0B72D7'
        elif "".join(map(str,stageSetup)) == '2103':
            lineOption += '#54B5FB'
    return lineOption 

    # # For manuscript cov10
    # lineOption = ''
    # if len(stageSetup) == 2:
    #     # pastel
    #     if "".join(map(str,stageSetup)) == '03':
    #         lineOption += '#F5821F'
    #     elif "".join(map(str,stageSetup)) == '13':
    #         lineOption += '#FDB827'
    #     elif "".join(map(str,stageSetup)) == '23':
    #         lineOption += '#61BB46'

    # if len(stageSetup) == 3:
    #     # First strong
    #     if "".join(map(str,stageSetup)) == '013':
    #         lineOption += '#00218C'
    #     elif "".join(map(str,stageSetup)) == '023':
    #         lineOption += '#2CB7D1'
    #     elif "".join(map(str,stageSetup)) == '103':
    #         lineOption += '#326633'
    #     elif "".join(map(str,stageSetup)) == '123':
    #         lineOption += '#00218C'
    #     elif "".join(map(str,stageSetup)) == '203':
    #         lineOption += '#094782'
    #     elif "".join(map(str,stageSetup)) == '213':
    #         lineOption += '#098BF5'
    
    # if len(stageSetup) == 4:
    #     if "".join(map(str,stageSetup)) == '0123':
    #         lineOption += '#B32E5F'
    #     elif "".join(map(str,stageSetup)) == '0213':
    #         lineOption += '#F8BCBD'
    #     elif "".join(map(str,stageSetup)) == '1023':
    #         lineOption += '#478F48'
    #     elif "".join(map(str,stageSetup)) == '1203':
    #         lineOption += '#71C0A7'
    #     elif "".join(map(str,stageSetup)) == '2013':
    #         lineOption += '#0B72D7'
    #     elif "".join(map(str,stageSetup)) == '2103':
    #         lineOption += '#54B5FB'
    # return lineOption 

def getLineColor2(stageSetup):
    # For manuscript cov10
    lineOption = ''
    if len(stageSetup) == 2:
        # pastel
        if "".join(map(str,stageSetup)) == '03':
            lineOption += '#F4BCF0'
        elif "".join(map(str,stageSetup)) == '13':
            lineOption += '#9272BC'
        elif "".join(map(str,stageSetup)) == '23':
            lineOption += '#8BCAEC'

    if len(stageSetup) == 3:
        # First strong
        if "".join(map(str,stageSetup)) == '013':
            lineOption += '#00218C'
        elif "".join(map(str,stageSetup)) == '023':
            lineOption += '#2CB7D1'
        elif "".join(map(str,stageSetup)) == '103':
            lineOption += '#326633'
        elif "".join(map(str,stageSetup)) == '123':
            lineOption += '#E5BC45'
        elif "".join(map(str,stageSetup)) == '203':
            lineOption += '#094782'
        elif "".join(map(str,stageSetup)) == '213':
            lineOption += '#098BF5'
    
    if len(stageSetup) == 4:
        if "".join(map(str,stageSetup)) == '0123':
            lineOption += '#B53647'
        elif "".join(map(str,stageSetup)) == '0213':
            lineOption += '#F8BCBD'
        elif "".join(map(str,stageSetup)) == '1023':
            lineOption += '#478F48'
        elif "".join(map(str,stageSetup)) == '1203':
            lineOption += '#71C0A7'
        elif "".join(map(str,stageSetup)) == '2013':
            lineOption += '#0B72D7'
        elif "".join(map(str,stageSetup)) == '2103':
            lineOption += '#54B5FB'
    return lineOption 

def drawMetricPerBudget(metric, allStages, restultPath, outPath, drawAll = True):
    fontP = FontProperties()
    fontP.set_size('xx-small')

    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    fileList = list()
    for root, dirs, files in os.walk(restultPath, topdown=True):
        for name in files:
            if not "minCost" in name:
                if not "DS_Store" in name:
                    fileList.append(root + name)
        break

    for fileName in fileList:
        with open(fileName, 'rb') as handle:
            data = pickle.load(handle)

        # with open(fileName.replace(".pickle", "_0.5_minCost.pickle"), 'rb') as handle:
        #     data_minCost = pickle.load(handle)

        if drawAll:
            stages = allStages[fileName]
        else:
            stage2Draw = allStages[fileName]
            checkFlag = [True]*10000
            stages = list()
            for setup in stage2Draw:
                if checkFlag[len(setup) - 1]:
                    stages.append(setup)
                    checkFlag[len(setup) - 1] = False

        maxDomain = []
        for stageSetup in stages:
            if len(data[''.join(map(str, stageSetup))]) > len(maxDomain):
                maxDomain = [d['x'] for d in data[''.join(map(str, stageSetup))]]

        dfList = list()
        algorithmList = []
        lStyle = {}
        
        # alternativeVal = int(len(maxDomain)/25)
        alternativeVal = 0
        if alternativeVal == 0: alternativeVal = 1
        for stageSetup in stages:
            for algorithm in data[''.join(map(str, stageSetup))][0]['algorithms']:
                if algorithm not in algorithmList:
                    algorithmList.append(algorithm)
                    dfList.append(pd.DataFrame(index=maxDomain[::alternativeVal]))
                dfs = data[''.join(map(str, stageSetup))]
                column2Add = list()
                for df in dfs:
                    if len(df) < 8:
                        column2Add.append([NaN]*len(stageSetup))
                    else:
                        if 'samples' == metric:
                            column2Add.append(df[algorithm + '_' + metric][:-1])
                        else:
                            column2Add.append(df[algorithm + '_' + metric])
                num2Add = len(maxDomain) - len(dfs)
                for i in range(num2Add):
                    column2Add.append([NaN]*len(stageSetup))
                column2Add = column2Add[::alternativeVal]
                column2Add = np.reshape(column2Add, (len(column2Add), len(stageSetup)))
                                
                for i in range(len(stageSetup)):
                    lineOption = ''
                    dfList[algorithmList.index(algorithm)]['S_' + ',S_'.join(map(str, stageSetup)) + ":" + str(stageSetup[i])] = column2Add[:,i]
                    if len(stageSetup) == 2:
                        if stageSetup[i] == 0:
                            lineOption += '#B73A3A'
                        elif stageSetup[i] == 1:
                            lineOption += '#EC5656'
                        elif stageSetup[i] == 2:
                            lineOption += '#F28A90'
                        elif stageSetup[i] == 3:
                            # lineOption += '#F8BCBD'
                            lineOption += 'k'
                    elif len(stageSetup) == 3:
                        if stageSetup[i] == 0:
                            lineOption += '#026645'
                        elif stageSetup[i] == 1:
                            lineOption += '#0AAC00'
                        elif stageSetup[i] == 2:
                            lineOption += '#23C26F'
                        elif stageSetup[i] == 3:
                            lineOption += 'k'
                    elif len(stageSetup) == 4:
                        if stageSetup[i] == 0:
                            lineOption += '#0B31A5'
                        elif stageSetup[i] == 1:
                            lineOption += '#0050EB'
                        elif stageSetup[i] == 2:
                            lineOption += '#0078ED'
                        elif stageSetup[i] == 3:
                            lineOption += 'k'
                    lStyle['S_' + ',S_'.join(map(str, stageSetup)) + ":" + str(stageSetup[i])] = lineOption
        if metric == 'lambdas':
            ylabel = 'Operator'
        elif metric == 'cost':
            ylabel = 'Computational cost consumed'
        elif metric == 'samples':
            ylabel = 'The number of samples to score'
        for i in range(len(algorithmList)):
            dfList[i].plot(figsize=(10, 8.5), style=lStyle, alpha=0.5)
            plt.title('Proposed approach')
            # plt.xlabel('Total computational budget $C$')
            plt.xlabel('Total resource budget')
            # plt.xlim(np.min(dfList[0].index)-1000, np.max(dfList[0].index)+1000)
            # plt.xlim(16000, 27000)
            plt.ylabel(ylabel)
            plt.grid(True)
            # plt.savefig(outPath + fileName.replace(restultPath, "").replace(".pickle", "") + algorithmList[i] + ".pdf")
            plt.close()
        

        stages = [[0, 3], [1, 2, 3], [0, 1, 2, 3]]
        for s_i in range(3):
            stageSetup = stages[s_i]
            maxDomain = []
            dfList = list()
            algorithmList = []
            lStyle = {}
            if len(data[''.join(map(str, stageSetup))]) > len(maxDomain):
                maxDomain = [d['x'] for d in data[''.join(map(str, stageSetup))]]

            ax = plt.subplot(3, 1, s_i + 1)
            alternativeVal = 0
            if alternativeVal == 0: alternativeVal = 1
            for algorithm in data[''.join(map(str, stageSetup))][0]['algorithms']:
                if algorithm not in algorithmList:
                    algorithmList.append(algorithm)
                    dfList.append(pd.DataFrame(index=maxDomain[::alternativeVal]))
                dfs = data[''.join(map(str, stageSetup))]
                column2Add = list()
                for df in dfs:
                    if len(df) < 8:
                        column2Add.append([NaN]*len(stageSetup))
                    else:
                        if 'samples' == metric:
                            column2Add.append(df[algorithm + '_' + metric][:-1])
                        else:
                            column2Add.append(df[algorithm + '_' + metric])
                num2Add = len(maxDomain) - len(dfs)
                for i in range(num2Add):
                    column2Add.append([NaN]*len(stageSetup))
                column2Add = column2Add[::alternativeVal]
                column2Add = np.reshape(column2Add, (len(column2Add), len(stageSetup)))
                                
                for i in range(len(stageSetup)):
                    lineOption = ''
                    if metric == 'lambdas':
                        dfList[algorithmList.index(algorithm)]['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + "]:\lambda_" + str(stageSetup[i] + 1) + '$'] = column2Add[:,i]
                    elif metric == 'cost':
                        dfList[algorithmList.index(algorithm)]['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + "]:\left|\mathbb{X}_" + str(stageSetup[i] + 1) + "\\right|\\times c_" + str(stageSetup[i] + 1) + '$'] = column2Add[:,i]
                    elif metric == 'samples':
                        dfList[algorithmList.index(algorithm)]['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + "]:\left|\mathbb{X}_" + str(stageSetup[i] + 1) + '\\right|$'] = column2Add[:,i]

                    # dfList[algorithmList.index(algorithm)]['$[S_' + ',S_'.join(map(str, stageSetup)) + "]:\lambda_" + str(stageSetup[i] + 1) + '$'] = column2Add[:,i]
                    
                    if len(stageSetup) == 2:
                        if stageSetup[i] == 0:
                            lineOption += '#CB7086'
                        elif stageSetup[i] == 1:
                            lineOption += '#4AB095'
                        elif stageSetup[i] == 2:
                            lineOption += '#4E7CC8'
                        elif stageSetup[i] == 3:
                            # lineOption += '#F8BCBD'
                            lineOption += 'k'
                    elif len(stageSetup) == 3:
                        if stageSetup[i] == 0:
                            lineOption += '#CB7086'
                        elif stageSetup[i] == 1:
                            lineOption += '#4AB095'
                        elif stageSetup[i] == 2:
                            lineOption += '#4E7CC8'
                        elif stageSetup[i] == 3:
                            lineOption += 'k'
                    elif len(stageSetup) == 4:
                        if stageSetup[i] == 0:
                            lineOption += '#CB7086'
                        elif stageSetup[i] == 1:
                            lineOption += '#4AB095'
                        elif stageSetup[i] == 2:
                            lineOption += '#4E7CC8'
                        elif stageSetup[i] == 3:
                            lineOption += 'k'
                    if metric == 'lambdas':
                        lStyle['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + "]:\lambda_" + str(stageSetup[i] + 1) + '$'] = lineOption
                    elif metric == 'cost':
                        lStyle['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + "]:\left|\mathbb{X}_" + str(stageSetup[i] + 1) + "\\right|\\times c_" + str(stageSetup[i] + 1) + '$'] = lineOption
                    elif metric == 'samples':
                        lStyle['$[S_' + ',S_'.join(map(str, np.array(stageSetup )+1)) + "]:\left|\mathbb{X}_" + str(stageSetup[i] + 1) + '\\right|$'] = lineOption
                    
            dfList[0].plot(figsize=(7.5, 6.5), style = lStyle, linewidth=0.8, alpha=0.7, ax=ax)
            # ax = dfList[1].plot(figsize=(7.5, 6.5), style = lStyle, linewidth=0.8, linestyle = 'dashed', alpha=0.7, ax=ax)
            # legend = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon = 1, fontsize = 'small', prop=font, ncol=2)
            legend = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon = 1, fontsize = 'small', prop=font, ncol=1)
            
            plt.xlim(0, 1*1000*(10**5))
            frame = legend.get_frame()
            frame.set_color('white')
            if s_i == 2:
                plt.xlabel('Total resource budget')
            if s_i == 1:
                if metric == 'lambdas':
                    plt.ylabel('Screening threshold')
                elif metric == 'cost':
                    plt.ylabel('Resources used by each stage')
                elif metric == 'samples':
                    plt.ylabel('The number of input samples at each stage')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(outPath + fileName.replace(restultPath, "").replace(".pickle", "") + "_" + metric + ".pdf", transparent=True)
        plt.close()        

if __name__ == "__main__":
	main()