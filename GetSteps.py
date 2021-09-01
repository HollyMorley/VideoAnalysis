from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import Utils
from Config import *
from pathlib import Path
from GetRuns import GetRuns
from scipy import stats
from glob import glob

class GetSteps:
    def __init__(self):
        super().__init__()

    def GetSteps(self,data):
        # create dataframes where either the RForepaw is present or the RHindpaw is present
        RForepawDF = data.drop(data[data.loc(axis=1)['RForepaw', 'likelihood'] < 0.95].index)
        RHindpawDF = data.drop(data[data.loc(axis=1)['RHindpaw', 'likelihood'] < 0.95].index)

        # add columns where define which frame step cycle starts and stops in (and when limb stops/stance phase)
        RForepawDF.loc(axis=1)['Start'] = np.nan
        RForepawDF.loc(axis=1)['Stop'] = np.nan
        RForepawDF.loc(axis=1)['End'] = np.nan

        RHindpawDF.loc(axis=1)['Start'] = np.nan
        RHindpawDF.loc(axis=1)['Stop'] = np.nan
        RHindpawDF.loc(axis=1)['End'] = np.nan

        # loop through runs
        for r in data.index.unique(level='Run'):
            # get start values for both legs. Start is when limb goes from stationary to moving
            startRForepaw = RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'][np.logical_and.reduce((
                        # difference in x position between next run and this run is bigger than 10
                        RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'].shift(-1) - RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'] > 10,
                        # difference in x position between this run and the last run is less than 10
                        RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'] - RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'].shift(1) < 10,
                        # difference in x position between last run and two runs away is less than 10
                        RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'].shift(1) - RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'].shift(2) < 10,
                        # difference in x position between two runs away and three runs away is less than 10
                        RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'].shift(2) - RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'].shift(3) < 10))]
            startRHindpaw = RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'][np.logical_and.reduce((
                        # difference in x position between next run and this run is bigger than 10
                        RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'].shift(-1) - RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'] > 10,
                        # difference in x position between this run and the last run is less than 10
                        RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'] - RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'].shift(1) < 10,
                        # difference in x position between last run and two runs away is less than 10
                        RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'].shift(1) - RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'].shift(2) < 10,
                        # difference in x position between two runs away and three runs away is less than 10
                        RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'].shift(2) - RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'].shift(3) < 10))]

            # get stop values for both limbs. Stop is where limb stops moving, NOT end of step cycle
            stopRForepaw = RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw','x'][np.logical_and.reduce((
                        # difference between this run and the last is more than 8
                        RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw','x'] - RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw','x'].shift(1) > 5,
                        # difference between next run and this run is less than 8
                        RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw','x'].shift(-1) - RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw','x'] < 5,
                        RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw','x'].shift(-2) - RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw','x'].shift(-1) < 5))]
            stopRHindpaw = RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw','x'][np.logical_and.reduce((
                        # difference between this run and the last is more than 8
                        RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw','x'] - RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw','x'].shift(1) > 5,
                        # difference between next run and this run is less than 8
                        RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw','x'].shift(-1) - RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw','x'] < 5,
                        RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw','x'].shift(-2) - RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw','x'].shift(-1) < 5))]

            # get end values for both limbs. End is found as the frame preceding the start values
            endALLRForepaw = list()
            endALLRHindpaw = list()
            for s in range(0, len(startRForepaw)):
                endRForepaw = RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'][
                    RForepawDF.loc(axis=0)[r].loc(axis=1)['RForepaw', 'x'].shift(-1) == startRForepaw.iloc[s]]
                endALLRForepaw.append(endRForepaw)
            for s in range(0, len(startRHindpaw)):
                endRHindpaw = RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'][
                    RHindpawDF.loc(axis=0)[r].loc(axis=1)['RHindpaw', 'x'].shift(-1) == startRHindpaw.iloc[s]]
                endALLRHindpaw.append(endRHindpaw)

            endRForepaw = pd.concat(endALLRForepaw)
            endRHindpaw = pd.concat(endALLRHindpaw)

            ## code for getting delta:
            # test = runs.loc(axis=0)[r].loc(axis=1)['RForepaw','x'][runs.loc(axis=0)[r].loc(axis=1)['RForepaw','likelihood']>0.95] - runs.loc(axis=0)[r].loc(axis=1)['RForepaw','x'].shift(1)[runs.loc(axis=0)[r].loc(axis=1)['RForepaw','likelihood']>0.95]
            # then....
            # mask finds points where direction changes...
            # mask = np.logical_or.reduce((test.shift(-1) - test > threshold, test.shift(-2) - test > threshold, test.shift(-3) - test > threshold, test.shift(-4) - test > threshold, test.shift(-5) - test > threshold))
            # plt.plot(test[mask],'x')
            #
            #test2 = test.loc[test[test < -10].index[-1]::][1::]
            #test3 = test2[test2 > 1]
            #test4 = test3[np.abs(test3.index - np.roll(test3.index,1)) > 10]

            # clean up end and start values by binning starts and ends to incomplete steps
            # delete last value from start values
            startRForepaw = startRForepaw[:-1:]
            startRHindpaw = startRHindpaw[:-1:]
            # delete first value from end values
            endRForepaw = endRForepaw[1::]
            endRHindpaw = endRHindpaw[1::]

            # ALTERNATIVE CODE FOR FINDING START AND END OF STEP
            # data = runs.loc(axis=0)[r].loc(axis=1)['RForepaw','x'][runs.loc(axis=0)[r].loc(axis=1)['RForepaw','likelihood']>0.95]
            # data2 = data - data.shift(1)
            # data4 = data2[data2 < 2]
            # data4 = data4.to_frame('data4')
            # data4.loc['Step'] = np.nan
            #
            # stepno = 0
            # for i in range(0,len(data4)):
            #     if data4.index[i] - data4.index[i-1] < 10:
            #         data4['Step'].iloc[i] = stepno
            #     else:
            #         stepno+=1
            #         data4['Step'].iloc[i] = stepno
            #
            # startALL = list()
            # endALL = list()
            # for i in data4['Step'].unique():
            #     start = data4['Step'][data4['Step'] == i].index[0]
            #     end = data4['Step'][data4['Step'] == i].index[-1]
            #     startALL.append(start)
            #     endALL.append(end)




            # label frames in dataframes for forepaw and hindpaw for each run
            for i in range(0,len(startRForepaw)):
                RForepawDF.loc(axis=0)[r,startRForepaw.index[i]]['Start'] = 1
            for i in range(0,len(startRHindpaw)):
                RHindpawDF.loc(axis=0)[r,startRHindpaw.index[i]]['Start'] = 1

            for i in range(0,len(endRForepaw)):
                RForepawDF.loc(axis=0)[r,endRForepaw.index[i]]['End'] = 1
            for i in range(0,len(endRHindpaw)):
                RHindpawDF.loc(axis=0)[r,endRHindpaw.index[i]]['End'] = 1

            for i in range(0,len(stopRForepaw)):
                RForepawDF.loc(axis=0)[r,stopRForepaw.index[i]]['Stop'] = 1
            for i in range(0,len(stopRHindpaw)):
                RHindpawDF.loc(axis=0)[r,stopRHindpaw.index[i]]['Stop'] = 1

        stepDFs = {
            'RForepaw': RForepawDF,
            'RHindpaw': RHindpawDF
        }

        return stepDFs


    def getStepInfo(self, RForepawDF, RHindpawDF):
        RForepawStepCycle =






    def Main(self,data):
        # *****below will be a for loop through all inputted data files once got general structure****
        runsdf = pd.read_hdf(data)