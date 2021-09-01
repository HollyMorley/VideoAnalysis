from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import utils
from Config import *
from pathlib import Path
from GetRuns import GetRuns
from scipy import stats
from glob import glob
from matplotlib.patches import Rectangle


def getFilepaths(data):
    filenameALL = list()
    skelfilenameALL = list()
    pathALL = list()
    for df in range(0, len(data)):
        filename = Path(data[df]).stem
        skelfilename = "%s_skeleton" %filename
        path = str(Path(data[df]).parent)
        filenameALL.append(filename)
        skelfilenameALL.append(skelfilename)
        pathALL.append(path)
    return filenameALL, skelfilenameALL, pathALL


def getCoorData(data):
    DataframeCoorALL = list()
    for df in range(0, len(data)):
        DataframeCoor = pd.read_hdf(data[df])
        DataframeCoorALL.append(DataframeCoor)
    return DataframeCoorALL

def getSkel(data):
    DataframeSkelALL = list()
    #DataframeCoorRunsALL = getCoorData(data)
    for df in range(0, len(data)):
        #path = DataframeCoorALL[df]
        #filename = Path(data[df]).stem.rsplit('_RunsAll')[0] #gets base name of files
        #filename = getFilepaths(data)[df][0]
        #skelefilename = "%s_skeleton.h5" %filename
        skelfilename = getFilepaths(data)[1][df]
        #pathname = str(Path(data[df]).parent)
        pathname = getFilepaths(data)[2][df]
        skelepath = "%s\\%s.h5" %(pathname, skelfilename)
        DataframeSkel = pd.read_hdf(skelepath)
        DataframeSkelALL.append(DataframeSkel)
    return DataframeSkelALL


def filterSkel(files, pcutoff=pcutoff):  # data should be the basic h5 file here (NOT RUNSALL FILE)
    # gets list of full paths for raw, original data
    dataRunsALL = list()
    dataALL = list()
    for l in range(0, len(files)):
        dataRuns = "%s\\%s_RunsAll.h5" %(getFilepaths(files)[2][l], getFilepaths(files)[0][l])
        dataRunsALL.append(dataRuns)
        data = "%s\\%s.h5" %(getFilepaths(files)[2][l], getFilepaths(files)[0][l])
        dataALL.append(data)
    # gets raw skeleton data
    DataframeSkelALL = getSkel(dataALL)
    # gets filtered coordinate data (only data from detected runs)
    DataframeCoorRunsALL = getCoorData(dataRunsALL)
    # gets filtered skeleton data (only data from detected runs)
    DataframeSkelRunsALL = list()
    for df in range(0, len(files)):
        DataframeCoor = GetRuns().getOriginalDF(files=[files[df]])
        mask = DataframeCoor.progress_apply(lambda x: GetRuns().getInFrame(x, pcutoff=pcutoff), axis=1)
        DataframeSkelRuns = DataframeSkelALL[df][mask]
        DataframeSkelRuns = DataframeSkelRuns.set_index([DataframeCoorRunsALL[df].index.get_level_values('Run'), DataframeCoorRunsALL[df].index.get_level_values('FrameIdx')])
        DataframeSkelRunsALL.append(DataframeSkelRuns)
    return DataframeSkelRunsALL

def getListOfSkeletons(files):
    DataframeSkelALL = getSkel(data= files)
    skeletonsColumns = DataframeSkelALL[0].columns.get_level_values(level=0)
    skeletonList = list(skeletonsColumns[::3])
    #print("Skeleton components available are: \n {}".format(skeletonList))
    return skeletonList

def getListOfMeasures(files, extraMeasure=None):
    DataframeSkelALL = getSkel(data=files)
    measureColumns = DataframeSkelALL[0].columns.get_level_values(level=1)
    measureList = list(measureColumns[0:2])
    if extraMeasure is not None:
        measureList.extend(extraMeasure)
    return measureList

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def plotSkel(files, skeletonList, measure, alphavalue=0.8, threshold=0.25, pcutoff=0.9, colourmap='cool', fs=(4,3)): # define skeleton list outside of function, could just be one skeleton
    ### Important information ###
    # files: list of basic h5 coordinate files
    # skeletonList: list of skeleton components you want to plot. Can write as str list, or run:
    #           skeletonList = getListOfSkeletons(files)
    #           skeletonList = skeletonList[2]
    # measure: either 'length' or 'orientation'
    # threshold: threshold for calculating outliers with z-scores
    # pcutoff: threshold for likelihood of plotted points
    # colourmap: examples incl 'tab20', 'tab20b', 'tab20c', 'tab10', 'Set3, 'gist_rainbow', 'gist_ncar', 'nipy_spectral'....

    DataframeSkelRunsALL = filterSkel(files)
    for df in range(0, len(files)):
        for skelIdx, skel in enumerate(skeletonList):
            plt.figure(figsize=fs)
            runs = DataframeSkelRunsALL[df].index.get_level_values(level=0).drop_duplicates().values.astype(int)
            z = np.abs(stats.zscore(DataframeSkelRunsALL[df].loc(axis=1)[skel, measure]))
            z = pd.DataFrame(z)
            z = z.set_index([DataframeSkelRunsALL[df].index.get_level_values('Run'), DataframeSkelRunsALL[df].index.get_level_values('FrameIdx')])
            for run in runs:
                colors = get_cmap(len(runs), name=colourmap)
                mask = np.logical_and(DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, 'likelihood'].values > pcutoff, z.loc(axis=0)[run][0].values < threshold)
                NormIdx = (DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].index - min(
                    DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].index)) / (
                                      max(DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].index) - min(
                                  DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].index))

                plt.plot(NormIdx[mask], DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].values[mask], '-', color=colors(run),alpha=alphavalue)
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colourmap), norm=plt.Normalize(vmin=0, vmax=len(runs) - 1))
            sm._A = []
            cbar = plt.colorbar(sm, ticks=range(len(runs)))
            cbar.set_ticklabels(runs)


def organiseSkelData(files, skeletonList, pcutoff, extraMeasure):
    DataframeRunMeanFiles = list()
    DataframeRunStdFiles = list()
    DataframeRunSemFiles = list()

    for df in range(0, len(files)):
        if type(files[df]) is str:
            files[df] = [files[df]]
        DataframeSkelRunsALL = filterSkel(files[df])

        # Check which day of experiments data is from and create run indexes which correspond to experimental phases
        exp = []
        runPhases = []
        indexList = []
        if '20201130' in files[df][0]:
            exp = 'APACharBaseline'
            runPhases = [list(range(0, 20))]
            indexList = ['BaselineRuns']
        elif '20201201' in files[df][0]:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 20))]
            indexList = ['BaselineRuns', 'APARuns']
        elif '20201202' in files[df][0]:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 20))]
            indexList = ['BaselineRuns', 'APARuns']
        elif '20201203' in files[df][0]:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 20))]
            indexList = ['BaselineRuns', 'APARuns']
        elif '20201204' in files[df][0]:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 20))]
            indexList = ['BaselineRuns', 'APARuns']
        elif '20201207' in files[df][0]:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
        elif '20201208' in files[df][0]:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
        elif '20201209' in files[df][0]:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
        elif '20201210' in files[df][0]:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
        elif '20201211' in files[df][0]:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
        elif '20201214' in files[df][0]:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
        elif '20201215' in files[df][0]:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
        elif '20201216' in files[df][0]:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
        elif '20201217' in files[df][0]:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
        else:
            print('Somethings gone wrong, cannot find this file')

        measure = getListOfMeasures(files=files[df], extraMeasure=extraMeasure)
        runs = DataframeSkelRunsALL[0].index.get_level_values(level=0).drop_duplicates().values.astype(int)
        discrete = np.around(np.linspace(0, 1, 101), 2)
        NormIdxALL = list()
        for run in runs:
            NormIdx = (DataframeSkelRunsALL[0].loc(axis=0)[run].index - min(DataframeSkelRunsALL[0].loc(axis=0)[run].index)) / (max(DataframeSkelRunsALL[0].loc(axis=0)[run].index) - min(DataframeSkelRunsALL[0].loc(axis=0)[run].index))
            NormIdxALL.append(NormIdx)

        # put normalised (0-1) index for each run into single long list and reset the df indexes to (original) 'Run' and (new) 'NormIdx'
        NormIdxALL = list(np.concatenate(NormIdxALL).flat)
        DataframeSkelRunsALL[0].loc(axis=1)['NormIdxShort'] = list(np.around(np.array(NormIdxALL), 2))
        DataframeSkelRunsALL[0].loc(axis=1)['NormIdx'] = NormIdxALL
        DataframeSkelRunsALL[0].loc(axis=1)['Run'] = DataframeSkelRunsALL[0].index.get_level_values(level=0)
        DataframeSkelRunsALL[0].set_index(['Run', 'NormIdxShort', 'NormIdx'], append=False, inplace=True)
        DataframeSkelRunsALL[0].loc(axis=1)['NormIdx'] = NormIdxALL

        # find the frames/rows which are closest to predetermined uniform discrete frame samples
        DataframeSkelRunsALLfiltered = list()
        for run in runs:
            discreteIdxALL = list()
            for n in discrete:
                discreteIdx = DataframeSkelRunsALL[0].loc(axis=0)[run].loc(axis=1)['NormIdx'].iloc[
                    (DataframeSkelRunsALL[0].loc(axis=0)[run].loc(axis=1)['NormIdx'] - n).abs().argsort()[
                    :1]].values[0]
                discreteIdxALL.append(discreteIdx)
            discreteIdxALL = list(dict.fromkeys(discreteIdxALL)) # removes duplicate discrete indexes, e.g. when there is a big gap in preserved frames, the closest value to a few of the discrete indexes will be the same value. These duplicates are removed and so there will be a gap in frames plotted here.

            # find frames/rows between the predetermined discrete frames and cut them from the dataframe
            cutFrames = DataframeSkelRunsALL[0].loc(axis=0)[run].index.get_level_values(
                level='NormIdx').difference(pd.Index(discreteIdxALL))
            x = DataframeSkelRunsALL[0].loc(axis=0)[run].drop(index=cutFrames, inplace=False, level='NormIdx')
            DataframeSkelRunsALLfiltered.append(x)
        DataframeSkelRunsALL[0] = pd.concat(DataframeSkelRunsALLfiltered, axis=0, keys=runs, names=['Run','NormIdxShort','NormIdx'])


        ### make new df for calculating stats across runs (within phases) IN ONE MOUSE or calculating avergaes at every time point ACROSS MICE
        columns = pd.MultiIndex.from_product([getListOfSkeletons(files[df]), ['length', 'orientation']],names=['skeletonList', 'measure'])
        if exp == 'APACharBaseline':
            index = pd.MultiIndex.from_product([['BaselineRuns', 'TotalRuns'], np.around(np.array(discrete), 2)], names=['Phase', 'NormIdxShort'])
        elif exp == 'APACharNoWash':
            index = pd.MultiIndex.from_product([['BaselineRuns', 'APARuns', 'TotalRuns'], np.around(np.array(discrete), 2)], names=['Phase', 'NormIdxShort'])
        elif exp == 'APAChar':
            index = pd.MultiIndex.from_product([['BaselineRuns', 'APARuns', 'WashoutRuns', 'TotalRuns'], np.around(np.array(discrete), 2)], names=['Phase', 'NormIdxShort'])
        elif exp == 'VisuoMotTransf':
            index = pd.MultiIndex.from_product([['BaselineRuns', 'VMTRuns', 'WashoutRuns', 'TotalRuns'], np.around(np.array(discrete), 2)], names=['Phase', 'NormIdxShort'])
        else:
            print('something gone wrong with experiment type assignment')

        DataframeRunMean = pd.DataFrame(data=None, index=index, columns=columns)
        DataframeRunStd = pd.DataFrame(data=None, index=index, columns=columns)
        DataframeRunSem = pd.DataFrame(data=None, index=index, columns=columns)

        for m in measure:
            if m == 'triangle height':
                # initialise new columns
                DataframeRunMean.loc(axis=1)['combined', m] = np.nan  # this is not a copy
                DataframeRunStd.loc(axis=1)['combined', m] = np.nan
                DataframeRunSem.loc(axis=1)['combined', m] = np.nan
            if m == 'triangle angles':
                # initialise new columns
                DataframeRunMean.loc(axis=1)['combined', 'nose angle'] = np.nan
                DataframeRunMean.loc(axis=1)['combined', 'hump angle'] = np.nan
                DataframeRunMean.loc(axis=1)['combined', 'tail1 angle'] = np.nan
                DataframeRunStd.loc(axis=1)['combined', 'nose angle'] = np.nan
                DataframeRunStd.loc(axis=1)['combined', 'hump angle'] = np.nan
                DataframeRunStd.loc(axis=1)['combined', 'tail1 angle'] = np.nan
                DataframeRunSem.loc(axis=1)['combined', 'nose angle'] = np.nan
                DataframeRunSem.loc(axis=1)['combined', 'hump angle'] = np.nan
                DataframeRunSem.loc(axis=1)['combined', 'tail1 angle'] = np.nan

            for d in discrete:
                if m == 'length' or m == 'orientation':
                    for skelIdx, skel in enumerate(skeletonList):
                        for phaseNo, phase in enumerate(indexList):
                            if all(item in list(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m].index.get_level_values(level=0).values) for item in runPhases[phaseNo]) == False:
                                todel = list(set(runPhases[phaseNo]).difference(list(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m].index.get_level_values(level=0).values)))
                                for i in range(0, len(todel)):
                                    runPhases[phaseNo].remove(todel[i])
                            DataframeRunMean.loc[(phase, d), (skel, m)] = np.mean(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff])
                            DataframeRunStd.loc[(phase, d), (skel, m)] = np.std(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff])
                            DataframeRunSem.loc[(phase, d), (skel, m)] = stats.sem(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff])

                        DataframeRunMean.loc[('TotalRuns', d), (skel, m)] = np.mean(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, 'likelihood'].values > pcutoff]) # averages across ALL runs for each time point across a single run
                        DataframeRunStd.loc[('TotalRuns', d), (skel, m)] = np.std(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, 'likelihood'].values > pcutoff])
                        DataframeRunSem.loc[('TotalRuns', d), (skel, m)] = stats.sem(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, 'likelihood'].values > pcutoff])

                if m == 'triangle height':
                    for phaseNo, phase in enumerate(indexList):
                        if all(item in list(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'length'].index.get_level_values(level=0).values) for item in runPhases[phaseNo]) == False:
                            todel = list(set(runPhases[phaseNo]).difference(list(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'length'].index.get_level_values(level=0).values)))
                            for i in range(0, len(todel)):
                                runPhases[phaseNo].remove(todel[i])
                        # use formula to get height of triangle on base a (Nose_Tail1) for phases
                        a = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'length'].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff]
                        b = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'length'].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff]
                        c = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'length'].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff]
                        s = 0.5*(a+b+c)
                        area = np.sqrt(s*(s-a)*(s-b)*(s-c))
                        height = 2*area/a
                        # assign height values (across runs) to df
                        DataframeRunMean.loc[(phase, d), ('combined', m)] = np.mean(height)
                        DataframeRunStd.loc[(phase, d), ('combined', m)] = np.std(height)
                        DataframeRunSem.loc[(phase, d), ('combined', m)] = stats.sem(height)


                    # use formula to get height of triangle on base a (Nose_Tail1) for total runs
                    a = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'length'][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'likelihood'].values > pcutoff]
                    b = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'length'][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'likelihood'].values > pcutoff]
                    c = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'length'][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'likelihood'].values > pcutoff]
                    s = 0.5*(a+b+c)
                    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
                    height = 2*area/a
                    # assign height values (across runs) to df
                    DataframeRunMean.loc[('TotalRuns', d), ('combined', m)] = np.mean(height)
                    DataframeRunStd.loc[('TotalRuns', d), ('combined', m)] = np.std(height)
                    DataframeRunSem.loc[('TotalRuns', d), ('combined', m)] = stats.sem(height)

                if m == 'triangle angles':
                    for phaseNo, phase in enumerate(indexList):
                        if all(item in list(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'orientation'].index.get_level_values(level=0).values) for item in runPhases[phaseNo]) == False:
                            todel = list(set(runPhases[phaseNo]).difference(list(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'orientation'].index.get_level_values(level=0).values)))
                            for i in range(0, len(todel)):
                                runPhases[phaseNo].remove(todel[i])
                        # use formula to get angles of triangle for phases
                        nose_hump = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'orientation'].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff]
                        nose_tail1 = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'orientation'].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff]
                        hump_tail1 = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'orientation'].loc(axis=0)[runPhases[phaseNo]][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'likelihood'].loc(axis=0)[runPhases[phaseNo]].values > pcutoff]

                        angleNose = nose_hump - nose_tail1
                        angleTail1 = nose_tail1 - hump_tail1
                        angleHump = 180 - (angleNose + angleTail1)

                        # assign angle values (across runs) to df
                        DataframeRunMean.loc[(phase, d), ('combined', 'nose angle')] = np.mean(angleNose)
                        DataframeRunMean.loc[(phase, d), ('combined', 'hump angle')] = np.mean(angleHump)
                        DataframeRunMean.loc[(phase, d), ('combined', 'tail1 angle')] = np.mean(angleTail1)

                        DataframeRunStd.loc[(phase, d), ('combined', 'nose angle')] = np.std(angleNose)
                        DataframeRunStd.loc[(phase, d), ('combined', 'hump angle')] = np.std(angleHump)
                        DataframeRunStd.loc[(phase, d), ('combined', 'tail1 angle')] = np.std(angleTail1)
                        DataframeRunSem.loc[(phase, d), ('combined', 'nose angle')] = stats.sem(angleNose)
                        DataframeRunSem.loc[(phase, d), ('combined', 'hump angle')] = stats.sem(angleHump)
                        DataframeRunSem.loc[(phase, d), ('combined', 'tail1 angle')] = stats.sem(angleTail1)

                     # use formula to get angles of triangle for total runs
                    nose_hump = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'orientation'][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'likelihood'].values > pcutoff]
                    nose_tail1 = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'orientation'][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'likelihood'].values > pcutoff]
                    hump_tail1 = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'orientation'][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'likelihood'].values > pcutoff]

                    angleNose = nose_hump - nose_tail1
                    angleTail1 = nose_tail1 - hump_tail1
                    angleHump = 180 - (angleNose + angleTail1)

                    # assign angle values (across runs) to df
                    DataframeRunMean.loc[('TotalRuns', d), ('combined', 'nose angle')] = np.mean(angleNose)
                    DataframeRunMean.loc[('TotalRuns', d), ('combined', 'hump angle')] = np.mean(angleHump)
                    DataframeRunMean.loc[('TotalRuns', d), ('combined', 'tail1 angle')] = np.mean(angleTail1)

                    DataframeRunStd.loc[('TotalRuns', d), ('combined', 'nose angle')] = np.std(angleNose)
                    DataframeRunStd.loc[('TotalRuns', d), ('combined', 'hump angle')] = np.std(angleHump)
                    DataframeRunStd.loc[('TotalRuns', d), ('combined', 'tail1 angle')] = np.std(angleTail1)

                    DataframeRunSem.loc[('TotalRuns', d), ('combined', 'nose angle')] = stats.sem(angleNose)
                    DataframeRunSem.loc[('TotalRuns', d), ('combined', 'hump angle')] = stats.sem(angleHump)
                    DataframeRunSem.loc[('TotalRuns', d), ('combined', 'tail1 angle')] = stats.sem(angleTail1)

        # collect dfs for each video into a single list for mean and std
        DataframeRunMeanFiles.append(DataframeRunMean)
        DataframeRunStdFiles.append(DataframeRunStd)
        DataframeRunSemFiles.append(DataframeRunSem)
    return DataframeRunMeanFiles, DataframeRunStdFiles, DataframeRunSemFiles

def organiseSkelDataByRuns(files, skeletonList, pcutoff, extraMeasure):
    DataframeRunMeanFiles = list()
    DataframeRunStdFiles = list()
    DataframeRunSemFiles = list()

    for df in range(0, len(files)):
        if type(files[df]) is str:
            files[df] = [files[df]]
        DataframeSkelRunsALL = filterSkel(files[df])

        measure = getListOfMeasures(files[df], extraMeasure=extraMeasure)
        runs = DataframeSkelRunsALL[0].index.get_level_values(level=0).drop_duplicates().values.astype(int)
        discrete = np.around(np.linspace(0, 1, 101), 2)
        NormIdxALL = list()
        for run in runs:
            NormIdx = (DataframeSkelRunsALL[0].loc(axis=0)[run].index - min(DataframeSkelRunsALL[0].loc(axis=0)[run].index)) / (max(DataframeSkelRunsALL[0].loc(axis=0)[run].index) - min(DataframeSkelRunsALL[0].loc(axis=0)[run].index))
            NormIdxALL.append(NormIdx)

        # put normalised (0-1) index for each run into single long list and reset the df indexes to (original) 'Run' and (new) 'NormIdx'
        NormIdxALL = list(np.concatenate(NormIdxALL).flat)
        DataframeSkelRunsALL[0].loc(axis=1)['NormIdxShort'] = list(np.around(np.array(NormIdxALL), 2))
        DataframeSkelRunsALL[0].loc(axis=1)['NormIdx'] = NormIdxALL
        DataframeSkelRunsALL[0].loc(axis=1)['Run'] = DataframeSkelRunsALL[0].index.get_level_values(level=0)
        DataframeSkelRunsALL[0].set_index(['Run', 'NormIdxShort', 'NormIdx'], append=False, inplace=True)
        DataframeSkelRunsALL[0].loc(axis=1)['NormIdx'] = NormIdxALL

        # find the frames/rows which are closest to predetermined uniform discrete frame samples
        DataframeSkelRunsALLfiltered = list()
        for run in runs:
            discreteIdxALL = list()
            for n in discrete:
                discreteIdx = DataframeSkelRunsALL[0].loc(axis=0)[run].loc(axis=1)['NormIdx'].iloc[
                    (DataframeSkelRunsALL[0].loc(axis=0)[run].loc(axis=1)['NormIdx'] - n).abs().argsort()[
                    :1]].values[0]
                discreteIdxALL.append(discreteIdx)
            discreteIdxALL = list(dict.fromkeys(discreteIdxALL)) # removes duplicate discrete indexes, e.g. when there is a big gap in preserved frames, the closest value to a few of the discrete indexes will be the same value. These duplicates are removed and so there will be a gap in frames plotted here.

            # find frames/rows between the predetermined discrete frames and cut them from the dataframe
            cutFrames = DataframeSkelRunsALL[0].loc(axis=0)[run].index.get_level_values(
                level='NormIdx').difference(pd.Index(discreteIdxALL))
            x = DataframeSkelRunsALL[0].loc(axis=0)[run].drop(index=cutFrames, inplace=False, level='NormIdx')
            DataframeSkelRunsALLfiltered.append(x)
        DataframeSkelRunsALL[0] = pd.concat(DataframeSkelRunsALLfiltered, axis=0, keys=runs, names=['Run','NormIdxShort','NormIdx'])

        endTimePts = discrete[-25:] # get the discrete frame indexes from the final 1/4 of the run

        ### make new df for calculating stats within runs
        columns = pd.MultiIndex.from_product([MakeGraphs().getListOfSkeletons(files[df]), ['length', 'orientation']],names=['skeletonList', 'measure'])
        index = pd.MultiIndex.from_product([runs], names=['Run'])

        DataframeRunMean = pd.DataFrame(data=None, index=index, columns=columns)
        DataframeRunStd = pd.DataFrame(data=None, index=index, columns=columns)
        DataframeRunSem = pd.DataFrame(data=None, index=index, columns=columns)

        for m in measure:
            if m == 'triangle height':
                # initialise new columns
                DataframeRunMean.loc(axis=1)['combined', m] = np.nan
                DataframeRunStd.loc(axis=1)['combined', m] = np.nan
                DataframeRunSem.loc(axis=1)['combined', m] = np.nan
            if m == 'triangle angles':
                # initialise new columns
                DataframeRunMean.loc(axis=1)['combined', 'nose angle'] = np.nan
                DataframeRunMean.loc(axis=1)['combined', 'hump angle'] = np.nan
                DataframeRunMean.loc(axis=1)['combined', 'tail1 angle'] = np.nan
                DataframeRunStd.loc(axis=1)['combined', 'nose angle'] = np.nan
                DataframeRunStd.loc(axis=1)['combined', 'hump angle'] = np.nan
                DataframeRunStd.loc(axis=1)['combined', 'tail1 angle'] = np.nan
                DataframeRunSem.loc(axis=1)['combined', 'nose angle'] = np.nan
                DataframeRunSem.loc(axis=1)['combined', 'hump angle'] = np.nan
                DataframeRunSem.loc(axis=1)['combined', 'tail1 angle'] = np.nan

            for run in runs:
                if m == 'length' or m == 'orientation':
                    for skelIdx, skel in enumerate(skeletonList):
                        DataframeRunMean.loc[run, (skel, m)] = np.mean(DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel,m)][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel,'likelihood')].values > pcutoff])
                        DataframeRunStd.loc[run, (skel, m)] = np.std(DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel,m)][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel,'likelihood')].values > pcutoff])
                        DataframeRunSem.loc[run, (skel, m)] = stats.sem(DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel,m)][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel,'likelihood')].values > pcutoff])

                if m == 'triangle height':
                    # use formula to get height of triangle on base a (Nose_Tail1) for phases
                    a = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Tail1', 'length')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Tail1', 'likelihood')].values > pcutoff]
                    b = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Hump', 'length')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Hump', 'likelihood')].values > pcutoff]
                    c = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Hump_Tail1', 'length')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Hump_Tail1', 'likelihood')].values > pcutoff]
                    s = 0.5*(a+b+c)
                    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
                    height = 2*area/a
                    # assign height values (across runs) to df
                    DataframeRunMean.loc[run, ('combined', m)] = np.mean(height)
                    DataframeRunStd.loc[run, ('combined', m)] = np.std(height)
                    DataframeRunSem.loc[run, ('combined', m)] = stats.sem(height)

                if m == 'triangle angles':
                    # use formula to get angles of triangle for phases
                    nose_hump = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Hump', 'orientation')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Hump', 'likelihood')].values > pcutoff]
                    nose_tail1 = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Tail1', 'orientation')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Tail1', 'likelihood')].values > pcutoff]
                    hump_tail1 = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Hump_Tail1', 'orientation')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Hump_Tail1', 'likelihood')].values > pcutoff]

                    angleNose = nose_hump - nose_tail1
                    angleTail1 = nose_tail1 - hump_tail1
                    angleHump = 180 - (angleNose + angleTail1)

                    # assign angle values (across runs) to df
                    DataframeRunMean.loc[run, ('combined', 'nose angle')] = np.mean(angleNose)
                    DataframeRunMean.loc[run, ('combined', 'hump angle')] = np.mean(angleHump)
                    DataframeRunMean.loc[run, ('combined', 'tail1 angle')] = np.mean(angleTail1)

                    DataframeRunStd.loc[run, ('combined', 'nose angle')] = np.std(angleNose)
                    DataframeRunStd.loc[run, ('combined', 'hump angle')] = np.std(angleHump)
                    DataframeRunStd.loc[run, ('combined', 'tail1 angle')] = np.std(angleTail1)

                    DataframeRunSem.loc[run, ('combined', 'nose angle')] = stats.sem(angleNose)
                    DataframeRunSem.loc[run, ('combined', 'hump angle')] = stats.sem(angleHump)
                    DataframeRunSem.loc[run, ('combined', 'tail1 angle')] = stats.sem(angleTail1)

        # collect dfs for each video into a single list for mean and std
        DataframeRunMeanFiles.append(DataframeRunMean)
        DataframeRunStdFiles.append(DataframeRunStd)
        DataframeRunSemFiles.append(DataframeRunSem)
    return DataframeRunMeanFiles, DataframeRunStdFiles, DataframeRunSemFiles


def saveSkelData(files=None, directory=None, destfolder=(), organisedby=(), extraMeasure=None, pcutoff=pcutoff):
    # type can be 'byRuns' or 'byTime' and refers to how data was organised
    # Default is to get all skeletons. For now, if only want a subset must run organiseSkelData() and save manually
    files = utils.Getlistoffiles(files=files, directory=directory)
    skeletonList = getListOfSkeletons(files=files)

    if organisedby == 'byTime':
        data = organiseSkelData(files=files, skeletonList=skeletonList, pcutoff=pcutoff, extraMeasure=extraMeasure)
        for l in range(0, len(data[0])):
            Meanfilename = "%s_SkelMean.h5" % Path(files[l][0]).stem
            Stdfilename = "%s_SkelStd.h5" % Path(files[l][0]).stem
            Semfilename = "%s_SkelSem.h5" % Path(files[l][0]).stem

            #save DataframeRunMeanFiles, DataframeRunStdFiles and DataframeRunSemFiles to file for each video
            data[0][l].to_hdf("%s\\%s" % (destfolder, Meanfilename), key='SkelMean', mode='a')
            print("Dataframe with mean skeleton values file saved for %s" % Path(files[l][0]).stem)
            data[1][l].to_hdf("%s\\%s" % (destfolder, Stdfilename), key='SkelStd', mode='a')
            print("Dataframe with standard deviations saved for %s" % Path(files[l][0]).stem)
            data[2][l].to_hdf("%s\\%s" % (destfolder, Semfilename), key='SkelSem', mode='a')
            print("Dataframe with standard error of the mean saved for %s" % Path(files[l][0]).stem)

    elif organisedby == 'byRuns':
        data = organiseSkelDataByRuns(files, skeletonList, pcutoff, extraMeasure)
        for l in range(0, len(data[0])):
            Meanfilename = "%s_SkelMean_byRuns.h5" % Path(files[l][0]).stem
            Stdfilename = "%s_SkelStd_byRuns.h5" % Path(files[l][0]).stem
            Semfilename = "%s_SkelSem_byRuns.h5" % Path(files[l][0]).stem

            # save DataframeRunMeanFiles, DataframeRunStdFiles and DataframeRunSemFiles to file for each video
            data[0][l].to_hdf("%s\\%s" % (destfolder, Meanfilename), key='SkelMeanbyRuns', mode='a')
            print("Dataframe with mean skeleton values file saved for %s" % Path(files[l][0]).stem)
            data[1][l].to_hdf("%s\\%s" % (destfolder, Stdfilename), key='SkelStdbyRuns', mode='a')
            print("Dataframe with standard deviations saved for %s" % Path(files[l][0]).stem)
            data[2][l].to_hdf("%s\\%s" % (destfolder, Semfilename), key='SkelSembyRuns', mode='a')
            print("Dataframe with standard error of the mean saved for %s" % Path(files[l][0]).stem)


def collateGroupSkelData(files, skeletonList, destfolder, organisedby):
    Mean = list()
    #DataframeSkelMeanGroup = list()

    # get all mice data together
    for df in range(0, len(files)):
        if type(files[df]) is str:
            files[df] = [files[df]]

        if organisedby == 'byTime':
            DataframeSkelMean = pd.read_hdf("%s\\%s_SkelMean.h5" % (destfolder, Path(files[df][0]).stem))
        elif organisedby == 'byRuns':
            DataframeSkelMean = pd.read_hdf("%s\\%s_SkelMean_byRuns.h5" % (destfolder, Path(files[df][0]).stem))

        # correct that all the skel columns are objects. Not perfect to do now but works for now.
        for skelIdx, skel in enumerate(skeletonList):
            for m in ['length', 'orientation']:
                DataframeSkelMean.loc(axis=1)[skel, m] = pd.to_numeric(DataframeSkelMean.loc(axis=1)[skel, m])
        Mean.append(DataframeSkelMean)
    DataframeSkelMeans = pd.DataFrame()
    for i in Mean:
        DataframeSkelMeans = DataframeSkelMeans.append(i)

    if organisedby == 'byTime':
        DataframeSkelMeanGroup = DataframeSkelMeans.groupby(level=[1,0]).mean()
        DataframeSkelStdGroup = DataframeSkelMeans.groupby(level=[1,0]).std()
        DataframeSkelSemGroup = DataframeSkelMeans.groupby(level=[1,0]).sem()
    if organisedby == 'byRuns':
        DataframeSkelMeanGroup = DataframeSkelMeans.groupby(level=0).mean()
        DataframeSkelStdGroup = DataframeSkelMeans.groupby(level=0).std()
        DataframeSkelSemGroup = DataframeSkelMeans.groupby(level=0).sem()

    return DataframeSkelMeanGroup, DataframeSkelStdGroup, DataframeSkelSemGroup


def saveSkelDataGroup(files=None, directory=None, destfolder=(), organisedby=()):
    files = utils.Getlistoffiles(files, directory)
    skeletonList = getListOfSkeletons(files)
    data = []

    if organisedby == 'byTime':
        data = collateGroupSkelData(files, skeletonList, destfolder, organisedby)

        Meanfilename = "%s_SkelMeanGroup.h5" % os.path.basename(destfolder)
        Stdfilename = "%s_SkelStdGroup.h5" % os.path.basename(destfolder)
        Semfilename = "%s_SkelSemGroup.h5" % os.path.basename(destfolder)

    elif organisedby == 'byRuns':
        data = collateGroupSkelData(files, skeletonList, destfolder, organisedby)

        Meanfilename = "%s_SkelMeanGroup_byRuns.h5" % os.path.basename(destfolder)
        Stdfilename = "%s_SkelStdGroup_byRuns.h5" % os.path.basename(destfolder)
        Semfilename = "%s_SkelSemGroup_byRuns.h5" % os.path.basename(destfolder)

    else:
        print('something went wrong with organisedby variable')

    #for l in range(0, len(data)):
    # save DataframeRunMeanFiles, DataframeRunStdFiles and DataframeRunSemFiles to file for each video
    data[0].to_hdf("%s\\%s" % (destfolder, Meanfilename), key='SkelMean', mode='a')
    print("Group Dataframe with mean skeleton values file saved for %s" % os.path.basename(destfolder))
    data[1].to_hdf("%s\\%s" % (destfolder, Stdfilename), key='SkelStd', mode='a')
    print("Group Dataframe with standard deviations saved for %s" % os.path.basename(destfolder))
    data[2].to_hdf("%s\\%s" % (destfolder, Semfilename), key='SkelSem', mode='a')
    print("Group Dataframe with standard error of the mean saved for %s" % os.path.basename(destfolder))

def getALLdata(directory=None, extraMeasure=None, pcutoff=pcutoff, organisedby=()):
    dirs = glob(os.path.join(directory, "*"))
    for l in range(0, len(dirs)):
        saveSkelData(directory=dirs[l], destfolder=dirs[l], organisedby=organisedby, extraMeasure=extraMeasure, pcutoff=pcutoff)
        print("Individual data saved for %s" % dirs[l])
        saveSkelDataGroup(directory=dirs[l], destfolder=dirs[l], organisedby=organisedby)
        print("Group data saved for %s" % dirs[l])
    print("Analysis finished!")

#     def plotAPAChar(self, directories, colormap='Pastel2'):
#         ################## put in skel, measure and 'fast_slow' or 'slow_fast' as a function input #####################
#         ################## each graph should have a line for each phase and each degree of the speed transition ##########
#         ###### input should be a list of the directories with APAChar files in them #########
#
#         allMeanbyTime = list()
#         allMeanbyRuns = list()
#         allStdbyTime = list()
#         allStdbyRuns = list()
#         allConditions = list()
#         allSpeeds = list()
#
#         skeletonList = ['Tail2_Tail3','Nose_Hump','Shoulder_Hump', 'Hip_RAnkle', 'Nose_Tail1', 'Hip_Tail1', 'Tail1_Tail2', 'Shoulder_Tail1', 'Hump_RAnkle', 'RHindpaw_RAnkle', 'Hump_Hip', 'Nose_Shoulder', 'Hump_RForepaw', 'Tail1_Tail3', 'Shoulder_RForepaw', 'Hump_Tail1']
#         measure = ['length', 'orientation', 'triangle height', 'triangle angles']
#         runs = list(range(1, 25))
#
#         for dir in range(0, len(directories)):
#             meanbyTime = pd.read_hdf("%s\\%s_SkelMeanGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
#             meanbyRuns = pd.read_hdf("%s\\%s_SkelMeanGroup_byRuns.h5" % (directories[dir], os.path.basename(directories[dir])))
#             stdbyTime = pd.read_hdf("%s\\%s_SkelStdGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
#             stdbyRuns = pd.read_hdf("%s\\%s_SkelStdGroup_byRuns.h5" % (directories[dir], os.path.basename(directories[dir])))
#
#             # Check which day of APA Characterise experiments data is from
#             condition = []
#             if os.path.basename(directories[dir]) == '20201130':
#                 condition = 'Control'
#                 speed = 0
#             elif os.path.basename(directories[dir]) == '20201201':
#                 condition = 'FastToSlow'
#                 speed = 0.08
#             elif os.path.basename(directories[dir]) == '20201204':
#                 condition = 'FastToSlow'
#                 speed = 0.04
#             elif os.path.basename(directories[dir]) == '20201208':
#                 condition = 'FastToSlow'
#                 speed = 0.16
#             elif os.path.basename(directories[dir]) == '20201209':
#                 condition = 'FastToSlow'
#                 speed = 0.32
#             elif os.path.basename(directories[dir]) == '20201202':
#                 condition = 'SlowToFast'
#                 speed = 0.04
#             elif os.path.basename(directories[dir]) == '20201203':
#                 condition = 'SlowToFast'
#                 speed = 0.16
#             elif os.path.basename(directories[dir]) == '20201207':
#                 condition = 'SlowToFast'
#                 speed = 0.08
#             elif os.path.basename(directories[dir]) == '20201210':
#                 condition = 'SlowToFast'
#                 speed = 0.32
#             elif os.path.basename(directories[dir]) == '20201211':
#                 condition = 'PerceptionTest'
#                 speed = 0.04
#             else:
#                 print('something wrong')
#
#             allMeanbyTime.append(meanbyTime)
#             allMeanbyRuns.append(meanbyRuns)
#             allStdbyTime.append(stdbyTime)
#             allStdbyRuns.append(stdbyRuns)
#             allConditions.append(condition)
#             allSpeeds.append(speed)
#
#         ## plot 'byTime' data, ie plot line graphs of each measure and each skeleton for baseline, APA and washout
#         #################################
#
#         ## plot 'byRuns' data, ie on each plot there is a line graph with runs on x axis and all measures (seperately) on y axis. Do a seperate graph for each measure, skeleton and speed direction. Plot multiple speeds (of same direction) on a single plot.
#         for m in measure:
#             if m == 'length' or m == 'orientation':
#                 for skelIdx, skel in enumerate(skeletonList):
#                     fig = plt.figure(num="%s, %s" % (m, skel), figsize=(20, 15))
#                     ax = plt.subplot(111)
#                     ax.set_xlim(0, 25)
#                     # ax.set_ylim(0, 250)
#                     for conIdx, con in enumerate(allConditions):
#
#                         # Fast to Slow
#                         if con == 'FastToSlow':
#                             F2S = ax.title('Fast to Slow Transition')
#                             F2S = ax.plot(runs, allMeanbyRuns[conIdx].loc(axis=1)[skel, m], label=speed[conIdx])
#
#                         if con == 'SlowToFast':
#                             S2F = ax.title('Slow to Fast Transition')
#                             S2F = ax.plot(runs, allMeanbyRuns[conIdx].loc(axis=1)[skel, m], label=speed[conIdx])
#
#                     F2S =
#
#                     ####### SAVE #########
#
#             if m == 'triangle height':
#                 fig = plt.figure(num="%s" % (m), figsize=(20, 15))
#                 ax = plt.subplot(111)
#                 ax.set_xlim(0, 25)
#                 # ax.set_ylim(0, 250)
#                 for conIdx, con in enumerate(allConditions):
#
#                     # Fast to Slow
#                     if con == 'FastToSlow':
#                         F2S = ax.title('Fast to Slow Transition')
#                         F2S = ax.plot(runs, allMeanbyRuns[conIdx].loc(axis=1)[skel, m], label=speed[conIdx])
#
#                     if con == 'SlowToFast':
#                         S2F = ax.title('Slow to Fast Transition')
#                         S2F = ax.plot(runs, allMeanbyRuns[conIdx].loc(axis=1)[skel, m], label=speed[conIdx])
#
#
#
#         # get basic graph attributes
#         discrete = np.around(np.linspace(0, 1, 101), 2)
#         colors = MakeGraphs().get_cmap(n=2, name=colormap)
#
#         # get data for that group
#         mean = pd.read_hdf(r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201130\20201130_SkelMeanGroup.h5")
#         std = pd.read_hdf(r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201130\20201130_SkelStdGroup.h5")
#         sem = pd.read_hdf(r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201130\20201130_SkelSemGroup.h5")
#
#         fig = plt.figure(num=1, figsize=(20,15))
#         ax = plt.subplot(111)
#         ax.plot(discrete, mean.loc(axis=1)['combined', 'triangle height'].xs('TotalRuns', level='Phase'), color=colors(0), linewidth=4)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 250)
#         ax.fill_between(discrete,
#                          mean.loc(axis=1)['combined', 'triangle height'].xs('TotalRuns', level='Phase') - std.loc(axis=1)['combined', 'triangle height'].xs('TotalRuns', level='Phase'),
#                          mean.loc(axis=1)['combined', 'triangle height'].xs('TotalRuns', level='Phase') + std.loc(axis=1)['combined', 'triangle height'].xs('TotalRuns', level='Phase'),
#                          interpolate=True, alpha=0.2, color=colors(0))
#         Baselinebox = ax.add_patch(plt.Rectangle((0,-20), 0.2, 20, facecolor='blue',
#                               clip_on=False,linewidth = 0))
#         APAbox = ax.add_patch(plt.Rectangle((0.2,-20), 0.6, 20, facecolor='red',
#                               clip_on=False,linewidth = 0))
#         Washoutbox = ax.add_patch(plt.Rectangle((0.8, -20), 0.2, 20, facecolor='yellow',
#                                             clip_on=False, linewidth=0))
#
#         Baseline = ax.text(0.1, -10, r'$\mathregular{Baseline}$', fontsize=14, zorder=5,
#                         color='k')
#         plt.show()
#
#
#
#
#
# # for APA Char
# directories = [r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201130",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201201",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201202",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201203",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201204",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201207",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201208",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201209",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201210",
#                r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201211"]






# files = [r"C:\Users\Holly Morley\Documents\Documents\DualBelt\analysis\DLC_DualBelt\DualBelt_Side\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201202\HM-20201202FLR1034274_cam0_1DLC_resnet_50_DLC_DualBeltFeb11shuffle1_800000.h5"]
# skeletonList = getListOfSkeletons(files)
# skeletonList = skeletonList[2]
# if type(skeletonList) is str:
#     skeletonList = [skeletonList]



# def plotCoor(pcutoff, data=()):
#     DataframeCoor = getCoorData(data)
#     for l in range(0, len(data)):
#         for bpindex, bp in enumerate(bodyparts2plot):
#             bpPresentIdx = DataframeCoor[l].loc(axis=1)[bp, 'likelihood'] > pcutoff
#             plt.plot(DataframeCoor[l].loc(axis=1)[bp, 'x'].values[bpPresentIdx],
#                      DataframeCoor.loc(axis=1)[bp, 'y'].values[bpPresentIdx])



# bodyparts = DataframeCoor.columns.get_level_values(0) #you can read out the header to get body part names!
# # bodyparts2plot = bodyparts #you could also take a subset, i.e. =['snout']