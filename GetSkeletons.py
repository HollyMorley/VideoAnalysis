from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import Utils
from Config import *
from pathlib import Path
from NewGetRuns import GetRuns
from scipy import stats
from glob import glob
from matplotlib.patches import Rectangle

class GetSkeletons:
    def __init__(self):
        super().__init__()

    def getCoorData(self, data):
        DataframeCoorALL = list()
        for df in range(0, len(data)):
            DataframeCoor = pd.read_hdf(data[df])
            DataframeCoorALL.append(DataframeCoor)
        return DataframeCoorALL

    def getSkel(self, data):
        DataframeSkelALL = list()
        for df in range(0, len(data)):
            skelfilename = Utils().getFilepaths(data)[1][df]
            pathname = Utils().getFilepaths(data)[2][df]
            skelepath = "%s\\%s.h5" %(pathname, skelfilename)
            DataframeSkel = pd.read_hdf(skelepath)
            DataframeSkelALL.append(DataframeSkel)
        return DataframeSkelALL

    def filterSkel(self, files, pcutoff=pcutoff):  # data should be the basic h5 file here (NOT RUNSALL FILE)
        # gets list of full paths for raw, original data
        dataRunsALL = list()
        dataALL = list()
        for l in range(0, len(files)):
            dataRuns = "%s\\%s_RunsAll.h5" %(Utils().getFilepaths(files)[2][l], Utils().getFilepaths(files)[0][l])
            dataRunsALL.append(dataRuns)
            data = "%s\\%s.h5" %(Utils().getFilepaths(files)[2][l], Utils().getFilepaths(files)[0][l])
            dataALL.append(data)
        # gets raw skeleton data
        DataframeSkelALL = self.getSkel(dataALL)
        # gets filtered coordinate data (only data from detected runs)
        DataframeCoorRunsALL = self.getCoorData(dataRunsALL)
        # gets filtered skeleton data (only data from detected runs)
        DataframeSkelRunsALL = list()
        for df in range(0, len(files)):
            DataframeSkelRuns = DataframeSkelALL[df].loc(axis=0)[DataframeCoorRunsALL[df].index.get_level_values(level='FrameIdx')]
            DataframeSkelRuns = DataframeSkelRuns.set_index([DataframeCoorRunsALL[df].index.get_level_values('Run'), DataframeCoorRunsALL[df].index.get_level_values('FrameIdx')])
            DataframeSkelRunsALL.append(DataframeSkelRuns)
        return DataframeSkelRunsALL

    def getListOfSkeletons(self, files):
        DataframeSkelALL = self.getSkel(data= files)
        skeletonsColumns = DataframeSkelALL[0].columns.get_level_values(level=0)
        skeletonList = list(skeletonsColumns[::3])
        #print("Skeleton components available are: \n {}".format(skeletonList))
        return skeletonList

    def getListOfMeasures(self, files, extraMeasure=None):
        DataframeSkelALL = self.getSkel(data=files)
        measureColumns = DataframeSkelALL[0].columns.get_level_values(level=1)
        measureList = list(measureColumns[0:2])
        if extraMeasure is not None:
            measureList.extend(extraMeasure)
        return measureList

    def plotSkel(self, files, skeletonList, measure, alphavalue=0.8, threshold=0.25, pcutoff=0.9, colourmap='cool',
                 fs=(4, 3)):  # define skeleton list outside of function, could just be one skeleton
        ### Important information ###
        # files: list of basic h5 coordinate files
        # skeletonList: list of skeleton components you want to plot. Can write as str list, or run:
        #           skeletonList = getListOfSkeletons(files)
        #           skeletonList = skeletonList[2]
        # measure: either 'length' or 'orientation'
        # threshold: threshold for calculating outliers with z-scores
        # pcutoff: threshold for likelihood of plotted points
        # colourmap: examples incl 'tab20', 'tab20b', 'tab20c', 'tab10', 'Set3, 'gist_rainbow', 'gist_ncar', 'nipy_spectral'....

        DataframeSkelRunsALL = self.filterSkel(files)
        for df in range(0, len(files)):
            for skelIdx, skel in enumerate(skeletonList):
                plt.figure(figsize=fs)
                runs = DataframeSkelRunsALL[df].index.get_level_values(level=0).drop_duplicates().values.astype(int)
                z = np.abs(stats.zscore(DataframeSkelRunsALL[df].loc(axis=1)[skel, measure]))
                z = pd.DataFrame(z)
                z = z.set_index([DataframeSkelRunsALL[df].index.get_level_values('Run'),
                                 DataframeSkelRunsALL[df].index.get_level_values('FrameIdx')])
                for run in runs:
                    colors = Utils().get_cmap(len(runs), name=colourmap)
                    mask = np.logical_and(
                        DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, 'likelihood'].values > pcutoff,
                        z.loc(axis=0)[run][0].values < threshold)
                    NormIdx = (DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].index - min(
                        DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].index)) / (
                                      max(DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[
                                              skel, measure].index) - min(
                                  DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].index))

                    plt.plot(NormIdx[mask],
                             DataframeSkelRunsALL[df].loc(axis=0)[run].loc(axis=1)[skel, measure].values[mask], '-',
                             color=colors(run), alpha=alphavalue)
                sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colourmap),
                                           norm=plt.Normalize(vmin=0, vmax=len(runs) - 1))
                sm._A = []
                cbar = plt.colorbar(sm, ticks=range(len(runs)))
                cbar.set_ticklabels(runs)

    def organiseSkelData(self, files, skeletonList, pcutoff, extraMeasure):
        # arrange dataframe by time/frame
        DataframeRunMeanFiles = list()
        DataframeRunStdFiles = list()
        DataframeRunSemFiles = list()

        for df in range(0, len(files)):
            if type(files[df]) is str:
                files[df] = [files[df]]
            DataframeSkelRunsALL = self.filterSkel(files[df])

            # Check which day of experiments data is from and create run indexes which correspond to experimental phases
            ############################################################################################################
            ##################### THIS CAN NOW BE GOT FROM GET_EXP_DETAILS FUNCTION IN UTILS.PY ########################
            ############################################################################################################
            expdetails = Utils().get_exp_details(files[df][0])
            exp = expdetails['exp']
            runPhases = expdetails['runPhases']
            indexList = expdetails['indexList']

            measure = self.getListOfMeasures(files=files[df], extraMeasure=extraMeasure)
            runs = DataframeSkelRunsALL[0].index.get_level_values(level=0).drop_duplicates().values.astype(int)
            discrete = np.around(np.linspace(0, 1, 101), 2)
            NormIdxALL = list()
            for run in runs:
                NormIdx = (DataframeSkelRunsALL[0].loc(axis=0)[run].index - min(
                    DataframeSkelRunsALL[0].loc(axis=0)[run].index)) / (
                                      max(DataframeSkelRunsALL[0].loc(axis=0)[run].index) - min(
                                  DataframeSkelRunsALL[0].loc(axis=0)[run].index))
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
                discreteIdxALL = list(dict.fromkeys(
                    discreteIdxALL))  # removes duplicate discrete indexes, e.g. when there is a big gap in preserved frames, the closest value to a few of the discrete indexes will be the same value. These duplicates are removed and so there will be a gap in frames plotted here.

                # find frames/rows between the predetermined discrete frames and cut them from the dataframe
                cutFrames = DataframeSkelRunsALL[0].loc(axis=0)[run].index.get_level_values(
                    level='NormIdx').difference(pd.Index(discreteIdxALL))
                x = DataframeSkelRunsALL[0].loc(axis=0)[run].drop(index=cutFrames, inplace=False, level='NormIdx')
                DataframeSkelRunsALLfiltered.append(x)
            DataframeSkelRunsALL[0] = pd.concat(DataframeSkelRunsALLfiltered, axis=0, keys=runs,
                                                names=['Run', 'NormIdxShort', 'NormIdx'])

            ### make new df for calculating stats across runs (within phases) IN ONE MOUSE or calculating avergaes at every time point ACROSS MICE
            columns = pd.MultiIndex.from_product([self.getListOfSkeletons(files[df]), ['length', 'orientation']],
                                                 names=['skeletonList', 'measure'])
            if exp == 'APACharBaseline':
                index = pd.MultiIndex.from_product(
                    [['BaselineRuns', 'TotalRuns'], np.around(np.array(discrete), 2)],
                    names=['Phase', 'NormIdxShort'])
            elif exp == 'APACharNoWash':
                index = pd.MultiIndex.from_product(
                    [['BaselineRuns', 'APARuns', 'TotalRuns'], np.around(np.array(discrete), 2)],
                    names=['Phase', 'NormIdxShort'])
            elif exp == 'APAChar':
                index = pd.MultiIndex.from_product(
                    [['BaselineRuns', 'APARuns', 'WashoutRuns', 'TotalRuns'], np.around(np.array(discrete), 2)],
                    names=['Phase', 'NormIdxShort'])
            elif exp == 'VisuoMotTransf':
                index = pd.MultiIndex.from_product(
                    [['BaselineRuns', 'VMTRuns', 'WashoutRuns', 'TotalRuns'], np.around(np.array(discrete), 2)],
                    names=['Phase', 'NormIdxShort'])
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

                            DataframeRunMean.loc[('TotalRuns', d), (skel, m)] = np.mean(DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, m][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)[skel, 'likelihood'].values > pcutoff])  # averages across ALL runs for each time point across a single run
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
                            s = 0.5 * (a + b + c)
                            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                            height = 2 * area / a
                            # assign height values (across runs) to df
                            DataframeRunMean.loc[(phase, d), ('combined', m)] = np.mean(height)
                            DataframeRunStd.loc[(phase, d), ('combined', m)] = np.std(height)
                            DataframeRunSem.loc[(phase, d), ('combined', m)] = stats.sem(height)

                        # use formula to get height of triangle on base a (Nose_Tail1) for total runs
                        a = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'length'][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Tail1', 'likelihood'].values > pcutoff]
                        b = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'length'][DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Nose_Hump', 'likelihood'].values > pcutoff]
                        c = DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'length'][ DataframeSkelRunsALL[0].xs(d, level='NormIdxShort').loc(axis=1)['Hump_Tail1', 'likelihood'].values > pcutoff]
                        s = 0.5 * (a + b + c)
                        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                        height = 2 * area / a
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


    def organiseSkelDatabyZoneswithTime(self, files, skeletonList, pcutoff, extraMeasure):
        # reformat both skel data (using coordinate data) so that have quadrant info and time info (ie time spent, speed)
        DataframeZoneMeanFiles = list()
        DataframeZoneSemFiles = list()

        for df in range(0, len(files)):
            print("Processing file: %s" % files[df])
            if type(files[df]) is str:
                files[df] = [files[df]]
            DataframeSkelRuns = self.filterSkel(files[df])
            RunsAllfname = "%s\\%s_RunsAll.h5" %(Utils().getFilepaths(files[df])[2][0], Utils().getFilepaths(files[df])[0][0])
            Dataframe = pd.read_hdf(RunsAllfname)

            measure = self.getListOfMeasures(files=files[df], extraMeasure=extraMeasure)
            runs = DataframeSkelRuns[0].index.unique(level='Run')

            #### calculate triangle height and angles for each frame ###
            # calculate triangle heights
            a = DataframeSkelRuns[0].loc(axis=1)['Nose_Tail1', 'length'][DataframeSkelRuns[0].loc(axis=1)['Nose_Tail1', 'length'] > pcutoff]
            b = DataframeSkelRuns[0].loc(axis=1)['Nose_Hump', 'length'][DataframeSkelRuns[0].loc(axis=1)['Nose_Hump', 'length'] > pcutoff]
            c = DataframeSkelRuns[0].loc(axis=1)['Hump_Tail1', 'length'][DataframeSkelRuns[0].loc(axis=1)['Hump_Tail1', 'length'] > pcutoff]
            s = 0.5 * (a + b + c)
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            height = 2 * area / a
            DataframeSkelRuns[0].loc(axis=1)['combined', 'triangle height'] = height

            # calculate triangle angles
            nose_hump = DataframeSkelRuns[0].loc(axis=1)['Nose_Hump', 'orientation'][DataframeSkelRuns[0].loc(axis=1)['Nose_Hump', 'length'] > pcutoff]
            nose_tail1 = DataframeSkelRuns[0].loc(axis=1)['Nose_Tail1', 'orientation'][DataframeSkelRuns[0].loc(axis=1)['Nose_Tail1', 'length'] > pcutoff]
            hump_tail1 = DataframeSkelRuns[0].loc(axis=1)['Hump_Tail1', 'orientation'][DataframeSkelRuns[0].loc(axis=1)['Hump_Tail1', 'length'] > pcutoff]
            angleNose = nose_hump - nose_tail1
            angleTail1 = nose_tail1 - hump_tail1
            angleHump = 180 - (angleNose + angleTail1)
            DataframeSkelRuns[0].loc(axis=1)['combined', 'nose angle'] = angleNose
            DataframeSkelRuns[0].loc(axis=1)['combined', 'hump angle'] = angleHump
            DataframeSkelRuns[0].loc(axis=1)['combined', 'tail1 angle'] = angleTail1


            ### get mask for when mouse in each quadrant from coordinate dfs ###
            platformPos = np.mean(Dataframe.loc(axis=1)['Platform', 'x'][Dataframe.loc(axis=1)['Platform', 'likelihood'] > pcutoff])
            numQs = 4
            quadrants = list(range(1,numQs+1))
            qlength = platformPos/numQs
            Qx = list()
            for i in range(0,numQs+1):
                qx = i*qlength
                Qx.append(qx)
            InQ = list()
            for i in range(0,numQs):
                inQ = np.logical_and(Dataframe.loc(axis=1)['Tail1', 'x'] > Qx[i], Dataframe.loc(axis=1)['Tail1', 'x'] <= Qx[i+1]).values
                InQ.append(inQ)
            # add column with quadrant indexes
            DataframeSkelRuns[0].loc(axis=1)['Quadrants'] = np.nan
            for i in range(0,numQs):
                DataframeSkelRuns[0].loc[InQ[i], 'Quadrants'] = i+1

            ### get times and speeds ###
            # time spent
            #for r in runs:


            ### get means for measures in these quadrants ###
            index = pd.MultiIndex.from_product([runs,quadrants], names=["Run", "Quadrant"])
            DataframeSkelQuadsMean = pd.DataFrame(columns=DataframeSkelRuns[0].columns, index=index)
            DataframeSkelQuadsSEM = pd.DataFrame(columns=DataframeSkelRuns[0].columns, index=index)

            for r in runs:
                for q in quadrants:
                    DataframeSkelQuadsMean.loc(axis=0)[r].loc(axis=0)[q] = np.mean(DataframeSkelRuns[0].loc(axis=0)[r][DataframeSkelRuns[0].loc(axis=0)[r].loc(axis=1)['Quadrants'] == q])
                    DataframeSkelQuadsSEM.loc(axis=0)[r].loc(axis=0)[q] = stats.sem(DataframeSkelRuns[0].loc(axis=0)[r][DataframeSkelRuns[0].loc(axis=0)[r].loc(axis=1)['Quadrants'] == q])

            # Check which day of experiments data is from and create run indexes which correspond to experimental phases
            expdetails = Utils().get_exp_details(files[df][0])
            exp = expdetails['exp']

            if exp == 'VisuoMotTransf' or exp == 'APACharBaseline':
                runPhases = expdetails['runPhases']
                indexList = expdetails['indexList']
                baselinespeed = expdetails['AcBaselineSpeed']
                VMTspeed = expdetails['AcVMTSpeed']
                washoutspeed = expdetails['AcWashoutSpeed']
            else:
                runPhases = expdetails['splitrunPhases']
                indexList = expdetails['splitindexList']
                speed = expdetails['Acspeed']

            # add in time spent and speed in each quadrant for each run
            DataframeSkelQuadsMean.loc(axis=1)['TimeSpent (s)'] = np.nan
            DataframeSkelQuadsMean.loc(axis=1)['Speed (cm/s)'] = np.nan
            DataframeSkelQuadsSEM.loc(axis=1)['TimeSpent (s)'] = np.nan
            DataframeSkelQuadsSEM.loc(axis=1)['Speed (cm/s)'] = np.nan
            for r in runs:
                for q in quadrants:
                    try:
                        DataframeSkelQuadsMean.loc[(r,q), 'TimeSpent (s)'] = (np.array((DataframeSkelRuns[0].loc(axis=0)[r].loc(axis=1)['Quadrants'][DataframeSkelRuns[0].loc(axis=0)[r].loc(axis=1)['Quadrants'] == q].tail(1).index - DataframeSkelRuns[0].loc(axis=0)[r].loc(axis=1)['Quadrants'][DataframeSkelRuns[0].loc(axis=0)[r].loc(axis=1)['Quadrants'] == q].head(1).index))/sideFPS)[0]
                    except:
                        print("missing value")
                    try:
                        # subtract belt speed from overall recorded speed in runs where belt is on only
                        if exp == 'APACharNoWash' or exp == 'APAChar':
                            if r in runPhases[1] or r in runPhases[2]: # corresponds to the split APA trials
                                DataframeSkelQuadsMean.loc[(r,q), 'Speed (cm/s)'] = ((sideViewBeltLength/numQs)/DataframeSkelQuadsMean.loc[(r,q), 'TimeSpent (s)'][0]) - speed
                            else:
                                DataframeSkelQuadsMean.loc[(r,q), 'Speed (cm/s)'] = (sideViewBeltLength/numQs)/DataframeSkelQuadsMean.loc[(r,q), 'TimeSpent (s)'][0]
                        if exp == 'VisuoMotTransf':
                            if r in runPhases[0]:
                                DataframeSkelQuadsMean.loc[(r,q), 'Speed (cm/s)'] = ((sideViewBeltLength/numQs)/DataframeSkelQuadsMean.loc[(r,q), 'TimeSpent (s)'][0]) - baselinespeed
                            if r in runPhases[1]:
                                DataframeSkelQuadsMean.loc[(r,q), 'Speed (cm/s)'] = ((sideViewBeltLength/numQs)/DataframeSkelQuadsMean.loc[(r,q), 'TimeSpent (s)'][0]) - VMTspeed
                            if r in runPhases[2]:
                                DataframeSkelQuadsMean.loc[(r,q), 'Speed (cm/s)'] = ((sideViewBeltLength/numQs)/DataframeSkelQuadsMean.loc[(r,q), 'TimeSpent (s)'][0]) - washoutspeed
                        else:
                            DataframeSkelQuadsMean.loc[(r,q), 'Speed (cm/s)'] = (sideViewBeltLength/numQs)/DataframeSkelQuadsMean.loc[(r,q), 'TimeSpent (s)'][0]
                    except:
                        print("missing value")

            indexphase = pd.MultiIndex.from_product([indexList,quadrants], names=["Phase", "Quadrant"])
            DataframeSkelQuadsPhaseMean = pd.DataFrame(columns=DataframeSkelQuadsMean.columns, index=indexphase)
            DataframeSkelQuadsPhaseSEM = pd.DataFrame(columns=DataframeSkelQuadsMean.columns, index=indexphase)
            DataframeSkelQuadsMean = DataframeSkelQuadsMean.astype(float)

            for phaseno, phase in enumerate(indexList):
                for q in quadrants:
                    try:
                        DataframeSkelQuadsPhaseMean.loc[phase,q] = np.mean(DataframeSkelQuadsMean.loc(axis=0)[runPhases[phaseno]].xs(q,axis=0,level='Quadrant'))
                        DataframeSkelQuadsPhaseSEM.loc[phase,q] = stats.sem(DataframeSkelQuadsMean.loc(axis=0)[runPhases[phaseno]].xs(q,axis=0,level='Quadrant'))
                    except:
                        print('there is a wrong number of runs')
                        if runs[-1] >= runPhases[phaseno][0]:
                            runnumbers = list(range(runPhases[phaseno][0], int(DataframeSkelQuadsMean.index.unique(level='Run')[-1])+1))
                            DataframeSkelQuadsPhaseMean.loc[phase,q] = np.mean(DataframeSkelQuadsMean.loc(axis=0)[runnumbers].xs(q,axis=0,level='Quadrant'))
                            DataframeSkelQuadsPhaseSEM.loc[phase,q] = stats.sem(DataframeSkelQuadsMean.loc(axis=0)[runnumbers].xs(q,axis=0,level='Quadrant'))
                        else:
                            print("No runs in phase: %s" % phase)

            DataframeZoneMeanFiles.append(DataframeSkelQuadsPhaseMean)
            DataframeZoneSemFiles.append(DataframeSkelQuadsPhaseSEM)

        return DataframeZoneMeanFiles, DataframeZoneSemFiles


    def organiseSkelDataByRuns(self, files, skeletonList, pcutoff, extraMeasure):
        # arrange dataframe by runs
        DataframeRunMeanFiles = list()
        DataframeRunStdFiles = list()
        DataframeRunSemFiles = list()

        for df in range(0, len(files)):
            if type(files[df]) is str:
                files[df] = [files[df]]
            DataframeSkelRunsALL = self.filterSkel(files[df])
            RunsAllfname = "%s\\%s_RunsAll.h5" % (Utils().getFilepaths(files[df])[2][0], Utils().getFilepaths(files[df])[0][0])
            Dataframe = pd.read_hdf(RunsAllfname)

            ### get mask for when mouse in each quadrant from coordinate dfs ###
            platformPos = np.mean(Dataframe.loc(axis=1)['Platform', 'x'][Dataframe.loc(axis=1)['Platform', 'likelihood'] > pcutoff])
            numQs = 4
            quadrants = list(range(1,numQs+1))
            qlength = platformPos/numQs
            Qx = list()
            for i in range(0,numQs+1):
                qx = i*qlength
                Qx.append(qx)
            InQ = list()
            for i in range(0,numQs):
                inQ = np.logical_and(Dataframe.loc(axis=1)['Tail1', 'x'] > Qx[i], Dataframe.loc(axis=1)['Tail1', 'x'] <= Qx[i+1]).values
                InQ.append(inQ)
            # add column with quadrant indexes
            DataframeSkelRunsALL[0].loc(axis=1)['Quadrants'] = np.nan
            for i in range(0,numQs):
                DataframeSkelRunsALL[0].loc[InQ[i], 'Quadrants'] = i+1

            measure = self.getListOfMeasures(files[df], extraMeasure=extraMeasure)
            runs = DataframeSkelRunsALL[0].index.get_level_values(level=0).drop_duplicates().values.astype(int)
            #discrete = np.around(np.linspace(0, 1, 301), 3)
            #discrete = np.arange(0.0,1.005,0.005) # gives 201 values
            #discrete = np.arange(0.0, 1.0025, 0.0025) # gives 401 values
            discrete = np.arange(0.0, 100001, 500) #gives 201 values
            NormIdxALL = list()
            for run in runs:
                NormIdx = (DataframeSkelRunsALL[0].loc(axis=0)[run].index - min(DataframeSkelRunsALL[0].loc(axis=0)[run].index)) / (max(DataframeSkelRunsALL[0].loc(axis=0)[run].index) - min(DataframeSkelRunsALL[0].loc(axis=0)[run].index))
                NormIdx = (NormIdx*100000).astype(int)
                NormIdxALL.append(NormIdx)

            # put normalised (0-1) index for each run into single long list and reset the df indexes to (original) 'Run' and (new) 'NormIdx'
            NormIdxALL = list(np.concatenate(NormIdxALL).flat)
           # DataframeSkelRunsALL[0].loc(axis=1)['NormIdxShort'] = list(np.around(np.array(NormIdxALL), 3))
            DataframeSkelRunsALL[0].loc(axis=1)['FrameIdx'] = DataframeSkelRunsALL[0].index.get_level_values(level='FrameIdx')
            DataframeSkelRunsALL[0].loc(axis=1)['NormIdx'] = NormIdxALL
            DataframeSkelRunsALL[0].loc(axis=1)['Run'] = DataframeSkelRunsALL[0].index.get_level_values(level=0)
            DataframeSkelRunsALL[0].set_index(['Run', 'FrameIdx', 'NormIdx'], append=False, inplace=True)
            DataframeSkelRunsALL[0].loc(axis=1)['NormIdx'] = NormIdxALL
            DataframeSkelRunsALL[0].loc(axis=1)['FrameIdx'] = DataframeSkelRunsALL[0].index.get_level_values(level='FrameIdx')

            # find the frames/rows which are closest to predetermined uniform discrete frame samples
            DataframeSkelRunsALLfiltered = list()
            for run in runs:
                discreteIdxALL = list()
                for n in discrete:
                    discreteIdx = DataframeSkelRunsALL[0].loc(axis=0)[run].loc(axis=1)['NormIdx'].iloc[(DataframeSkelRunsALL[0].loc(axis=0)[run].loc(axis=1)['NormIdx'] - n).abs().argsort()[:1]].values[0]
                    discreteIdxALL.append(discreteIdx)
                repeatmask = abs(np.roll(discreteIdxALL, -1) - discreteIdxALL) > 0
                discreteIdxALL = list(dict.fromkeys(discreteIdxALL))  # removes duplicate discrete indexes, e.g. when there is a big gap in preserved frames, the closest value to a few of the discrete indexes will be the same value. These duplicates are removed and so there will be a gap in frames plotted here.
                # find frames/rows between the predetermined discrete frames and cut them from the dataframe
                cutFrames = DataframeSkelRunsALL[0].loc(axis=0)[run].index.get_level_values(level='NormIdx').difference(pd.Index(discreteIdxALL))
                x = DataframeSkelRunsALL[0].loc(axis=0)[run].drop(index=cutFrames, inplace=False, level='NormIdx')
                x.loc(axis=1)['discreteIdx'] = discrete[repeatmask]
                x.loc(axis=1)['FrameIdx'] = DataframeSkelRunsALL[0].loc(axis=0)[run].loc(axis=1)['FrameIdx'][DataframeSkelRunsALL[0].loc(axis=0)[run].loc(axis=1)['NormIdx'].isin(discreteIdxALL)]
                x.set_index(['discreteIdx','FrameIdx'], append=False,inplace=True)
                DataframeSkelRunsALLfiltered.append(x)
            DataframeSkelRunsALL[0] = pd.concat(DataframeSkelRunsALLfiltered, axis=0, keys=runs,names=['Run', 'discreteIdx','FrameIdx'])
            DataframeSkelRunsALL[0].loc(axis=1)['FrameIdx'] = DataframeSkelRunsALL[0].index.get_level_values(level='FrameIdx')

            endTimePts = discrete  # get the discrete frame indexes from the final half of the run (including where nose not present)

            ### make new df for calculating stats within runs
            columns = pd.MultiIndex.from_product([self.getListOfSkeletons(files[df]), ['length', 'orientation']],
                                                 names=['skeletonList', 'measure'])
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
                    noseinframe = DataframeSkelRunsALL[0].loc(axis=0)[run].index.get_level_values(level='FrameIdx').isin(Dataframe.loc(axis=0)[run].index[Dataframe.loc(axis=0)[run].loc(axis=1)['Nose','likelihood'] > pcutoff]) # creates mask for the filtered frames which correspond to frames when the nose is in frame
                    endTimePts = discrete[np.in1d(discrete,np.array(DataframeSkelRunsALL[0].xs(run,level='Run').index.get_level_values(level='discreteIdx'))[noseinframe])]
                    #if len(DataframeSkelRunsALL[0].xs(run, level='Run').index[DataframeSkelRunsALL[0].xs(run, level='Run').index >= endTimePts[0]]) == len(endTimePts):
                    # if len(DataframeSkelRunsALL[0].xs(run, level='Run').loc(axis=0)[endTimePts]) == len(endTimePts):
                    #     endTimePts = endTimePts # do nothing
                    # else:
                    #     endTimePts = np.array(DataframeSkelRunsALL[0].xs(run, level='Run').loc(axis=0)[endTimePts].index.get_level_values(level='discreteIdx')[DataframeSkelRunsALL[0].xs(run, level='Run').loc(axis=0)[endTimePts].index.get_level_values(level='discreteIdx') >= endTimePts[0]])

                    if m == 'length' or m == 'orientation':
                        for skelIdx, skel in enumerate(skeletonList):
                            DataframeRunMean.loc[run, (skel, m)] = np.mean(DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel, m)][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel, 'likelihood')].values > pcutoff])
                            DataframeRunStd.loc[run, (skel, m)] = np.std(DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel, m)][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel, 'likelihood')].values > pcutoff])
                            DataframeRunSem.loc[run, (skel, m)] = stats.sem(DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel, m)][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, (skel, 'likelihood')].values > pcutoff])

                    if m == 'triangle height':
                        # use formula to get height of triangle on base a (Nose_Tail1) for phases
                        a = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Tail1', 'length')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Tail1', 'likelihood')].values > pcutoff]
                        b = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Hump', 'length')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Nose_Hump', 'likelihood')].values > pcutoff]
                        c = DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Hump_Tail1', 'length')][DataframeSkelRunsALL[0].xs(run, level='Run').loc[endTimePts, ('Hump_Tail1', 'likelihood')].values > pcutoff]
                        s = 0.5 * (a + b + c)
                        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                        height = 2 * area / a
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

    def saveSkelData(self, files=None, directory=None, destfolder=(), organisedby=(), extraMeasure=None,
                     pcutoff=pcutoff):
        # type can be 'byRuns' or 'byTime' and refers to how data was organised
        # Default is to get all skeletons. For now, if only want a subset must run organiseSkelData() and save manually
        if '20201218' in directory:
            files = Utils().Getlistoffiles(files=files, directory=directory,scorer=diffscorer)
        else:
            files = Utils().Getlistoffiles(files=files, directory=directory,scorer=scorer)
        skeletonList = self.getListOfSkeletons(files=files)

        if organisedby == 'byTime':
            data = self.organiseSkelData(files=files, skeletonList=skeletonList, pcutoff=pcutoff,
                                         extraMeasure=extraMeasure)
            for l in range(0, len(data[0])):
                Meanfilename = "%s_SkelMean.h5" % Path(files[l][0]).stem
                Stdfilename = "%s_SkelStd.h5" % Path(files[l][0]).stem
                Semfilename = "%s_SkelSem.h5" % Path(files[l][0]).stem

                # save DataframeRunMeanFiles, DataframeRunStdFiles and DataframeRunSemFiles to file for each video
                data[0][l].to_hdf("%s\\%s" % (destfolder, Meanfilename), key='SkelMean', mode='a')
                print("Dataframe with mean skeleton values file saved for %s" % Path(files[l][0]).stem)
                data[1][l].to_hdf("%s\\%s" % (destfolder, Stdfilename), key='SkelStd', mode='a')
                print("Dataframe with standard deviations saved for %s" % Path(files[l][0]).stem)
                data[2][l].to_hdf("%s\\%s" % (destfolder, Semfilename), key='SkelSem', mode='a')
                print("Dataframe with standard error of the mean saved for %s" % Path(files[l][0]).stem)

        elif organisedby == 'byRuns':
            data = self.organiseSkelDataByRuns(files, skeletonList, pcutoff, extraMeasure)
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

        elif organisedby == 'byZone':
            data = self.organiseSkelDatabyZoneswithTime(files,skeletonList,pcutoff,extraMeasure)
            for l in range(0, len(data[0])):
                Meanfilename = "%s_SkelMean_byZone.h5" % Path(files[l][0]).stem
                Semfilename = "%s_SkelSem_byZone.h5" % Path(files[l][0]).stem
                try:
                    # save DataframeRunMeanFiles, DataframeRunStdFiles and DataframeRunSemFiles to file for each video
                    data[0][l].to_hdf("%s\\%s" % (destfolder, Meanfilename), key='SkelMeanbyZone', mode='a')
                    print("Dataframe with mean skeleton values file saved for %s" % Path(files[l][0]).stem)
                    data[1][l].to_hdf("%s\\%s" % (destfolder, Semfilename), key='SkelSembyZone', mode='a')
                    print("Dataframe with standard error of the mean saved for %s" % Path(files[l][0]).stem)
                except:
                    print("couldn't save individual data")


    def collateGroupSkelData(self, files, skeletonList, destfolder, organisedby):
        Mean = list()
        # DataframeSkelMeanGroup = list()

        # get all mice data together
        for df in range(0, len(files)):
            if type(files[df]) is str:
                files[df] = [files[df]]

            if organisedby == 'byTime':
                DataframeSkelMean = pd.read_hdf("%s\\%s_SkelMean.h5" % (destfolder, Path(files[df][0]).stem))
            elif organisedby == 'byRuns':
                DataframeSkelMean = pd.read_hdf("%s\\%s_SkelMean_byRuns.h5" % (destfolder, Path(files[df][0]).stem))
            elif organisedby == 'byZone':
                DataframeSkelMean = pd.read_hdf("%s\\%s_SkelMean_byZone.h5" % (destfolder, Path(files[df][0]).stem))

            if organisedby == 'byTime' or organisedby == 'byRuns':
                # correct that all the skel columns are objects. Not perfect to do now but works for now.
                for skelIdx, skel in enumerate(skeletonList):
                    for m in ['length', 'orientation']:
                        DataframeSkelMean.loc(axis=1)[skel, m] = pd.to_numeric(DataframeSkelMean.loc(axis=1)[skel, m])
            if organisedby == 'byZone':
                for skelIdx, skel in enumerate(skeletonList):
                    for m in ['length', 'orientation']:
                        DataframeSkelMean.loc(axis=1)[skel, m] = pd.to_numeric(DataframeSkelMean.loc(axis=1)[skel, m])
                for comIdx, com in enumerate(['triangle height', 'nose angle', 'hump angle', 'tail1 angle']):
                    DataframeSkelMean.loc(axis=1)['combined', com] = pd.to_numeric(DataframeSkelMean.loc(axis=1)['combined', com])
                DataframeSkelMean.loc(axis=1)['Quadrants'] = pd.to_numeric(DataframeSkelMean.loc(axis=1)['Quadrants'])
                DataframeSkelMean.loc(axis=1)['TimeSpent (s)'] = pd.to_numeric(DataframeSkelMean.loc(axis=1)['TimeSpent (s)'])
                DataframeSkelMean.loc(axis=1)['Speed (cm/s)'] = pd.to_numeric(DataframeSkelMean.loc(axis=1)['Speed (cm/s)'])

            Mean.append(DataframeSkelMean)
        DataframeSkelMeans = pd.DataFrame()
        for i in Mean:
            DataframeSkelMeans = DataframeSkelMeans.append(i)

        if organisedby == 'byTime':
            DataframeSkelMeanGroup = DataframeSkelMeans.groupby(level=[1, 0]).mean()
            DataframeSkelStdGroup = DataframeSkelMeans.groupby(level=[1, 0]).std()
            DataframeSkelSemGroup = DataframeSkelMeans.groupby(level=[1, 0]).sem()
        if organisedby == 'byRuns':
            DataframeSkelMeanGroup = DataframeSkelMeans.groupby(level=0).mean()
            DataframeSkelStdGroup = DataframeSkelMeans.groupby(level=0).std()
            DataframeSkelSemGroup = DataframeSkelMeans.groupby(level=0).sem()
        if organisedby == 'byZone':
            DataframeSkelMeanGroup = DataframeSkelMeans.groupby(level=[0,1]).mean()
            DataframeSkelStdGroup = []
            DataframeSkelSemGroup = DataframeSkelMeans.groupby(level=[0,1]).sem()

        return DataframeSkelMeanGroup, DataframeSkelStdGroup, DataframeSkelSemGroup

    def saveSkelDataGroup(self, files=None, directory=None, destfolder=(), organisedby=()):
        if '20201218' in directory:
            files = Utils().Getlistoffiles(files, directory, diffscorer)
        else:
            files = Utils().Getlistoffiles(files, directory,scorer)
        skeletonList = self.getListOfSkeletons(files)
        data = []

        if organisedby == 'byTime':
            data = self.collateGroupSkelData(files, skeletonList, destfolder, organisedby)

            Meanfilename = "%s_SkelMeanGroup.h5" % os.path.basename(destfolder)
            Stdfilename = "%s_SkelStdGroup.h5" % os.path.basename(destfolder)
            Semfilename = "%s_SkelSemGroup.h5" % os.path.basename(destfolder)

        elif organisedby == 'byRuns':
            data = self.collateGroupSkelData(files, skeletonList, destfolder, organisedby)

            Meanfilename = "%s_SkelMeanGroup_byRuns.h5" % os.path.basename(destfolder)
            Stdfilename = "%s_SkelStdGroup_byRuns.h5" % os.path.basename(destfolder)
            Semfilename = "%s_SkelSemGroup_byRuns.h5" % os.path.basename(destfolder)

        elif organisedby == 'byZone':
            data = self.collateGroupSkelData(files, skeletonList, destfolder, organisedby)

            Meanfilename = "%s_SkelMeanGroup_byZone.h5" % os.path.basename(destfolder)
            Semfilename = "%s_SkelSemGroup_byZone.h5" % os.path.basename(destfolder)

        else:
            print('something went wrong with organisedby variable')

        if organisedby == 'byTime' or organisedby == 'byRuns':
            # save DataframeRunMeanFiles, DataframeRunStdFiles and DataframeRunSemFiles to file for each video
            data[0].to_hdf("%s\\%s" % (destfolder, Meanfilename), key='SkelMean', mode='a')
            print("Group Dataframe with mean skeleton values file saved for %s" % os.path.basename(destfolder))
            data[1].to_hdf("%s\\%s" % (destfolder, Stdfilename), key='SkelStd', mode='a')
            print("Group Dataframe with standard deviations saved for %s" % os.path.basename(destfolder))
            data[2].to_hdf("%s\\%s" % (destfolder, Semfilename), key='SkelSem', mode='a')
            print("Group Dataframe with standard error of the mean saved for %s" % os.path.basename(destfolder))
        elif organisedby == 'byZone':
            # save DataframeRunMeanFiles, DataframeRunStdFiles and DataframeRunSemFiles to file for each video
            data[0].to_hdf("%s\\%s" % (destfolder, Meanfilename), key='SkelMean', mode='a')
            print("Group Dataframe with mean skeleton values file saved for %s" % os.path.basename(destfolder))
            data[2].to_hdf("%s\\%s" % (destfolder, Semfilename), key='SkelSem', mode='a')
            print("Group Dataframe with standard error of the mean saved for %s" % os.path.basename(destfolder))

    def getALLdata(self, directory=None, extraMeasure=None, pcutoff=pcutoff, organisedby=()):
        dirs = glob(os.path.join(directory, "*"))
        for l in range(0, len(dirs)):
            self.saveSkelData(directory=dirs[l], destfolder=dirs[l], organisedby=organisedby,
                              extraMeasure=extraMeasure, pcutoff=pcutoff)
            print("Individual data saved for %s" % dirs[l])
            self.saveSkelDataGroup(directory=dirs[l], destfolder=dirs[l], organisedby=organisedby)
            print("Group data saved for %s" % dirs[l])
        print("Analysis finished!")


    def plotZone(self, directories, destfolder, colormap='cool'):
        #axis can either be quartiles or phase or None
        allMeanbyZone = list()
        allSembyZone = list()
        allConditions = list()
        allSpeeds = list()
        allPhases = list()

        skeletonList = ['Tail2_Tail3', 'Nose_Hump', 'Shoulder_Hump', 'Hip_RAnkle', 'Nose_Tail1', 'Hip_Tail1',
                        'Tail1_Tail2', 'Shoulder_Tail1', 'Hump_RAnkle', 'RHindpaw_RAnkle', 'Hump_Hip',
                        'Nose_Shoulder', 'Hump_RForepaw', 'Tail1_Tail3', 'Shoulder_RForepaw', 'Hump_Tail1']
        measure = ['length', 'orientation', 'triangle height', 'triangle angles', 'TimeSpent (s)', 'Speed (cm/s)']

        for dir in range(0, len(directories)):
            meanbyZone = pd.read_hdf(
                "%s\\%s_SkelMeanGroup_byZone.h5" % (directories[dir], os.path.basename(directories[dir])))
            sembyZone = pd.read_hdf(
                "%s\\%s_SkelSemGroup_byZone.h5" % (directories[dir], os.path.basename(directories[dir])))

            exp = Utils().get_exp_details(directories[dir])
            condition = exp['condition']
            speed = exp['Acspeed']
            perceivedspeed = exp['Pdspeed']
            phases = exp['splitindexList']

            allMeanbyZone.append(meanbyZone)
            allSembyZone.append(sembyZone)
            allConditions.append(condition)
            allSpeeds.append(speed)
            allPhases.append(phases)

        colors = Utils().get_cmap(n=4, name=colormap)
        patchcolors = Utils().get_cmap(n=6, name='Blues')
        for m in measure:
            if m == 'length' or m == 'orientation':
                for skelIdx, skel in enumerate(skeletonList):
                    # format F2S graph
                    F2S = plt.figure(figsize=(5.9,6.27)) # to change size of whole figure, when decide on dimensions i want, do F2S.get_figwidth() etc and then adjust accordingly
                    F2Swhole = F2S.add_subplot(111)
                    F2Sgs = F2S.add_gridspec(4, hspace=0)
                    F2Saxs = F2Sgs.subplots(sharex=True, sharey=True)
                    F2Saxs[-1].tick_params(top=False, bottom=False, left=True, right=False, labelbottom=False,labelcolor='black')
                    F2Swhole.spines['top'].set_color('none')
                    F2Swhole.spines['bottom'].set_color('none')
                    F2Swhole.spines['left'].set_color('none')
                    F2Swhole.spines['right'].set_color('none')
                    F2Swhole.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                    if m == 'length':
                        F2Swhole.set_ylabel('%s, %s (px)' % (skel,m), fontsize=16)
                    elif m == 'orientation':
                        F2Swhole.set_ylabel('%s, %s (°)' % (skel, m), fontsize=16)
                    F2Swhole.set_xlabel('Belt quadrant', fontsize=16)
                    F2S.suptitle('Belt 1 ON, Belt 2 OFF')

                    # format S2F graph
                    S2F = plt.figure(figsize=(5.9,6.27))  # to change size of whole figure, when decide on dimensions i want, do F2S.get_figwidth() etc and then adjust accordingly
                    S2Fwhole = S2F.add_subplot(111)
                    S2Fgs = S2F.add_gridspec(4, hspace=0)
                    S2Faxs = S2Fgs.subplots(sharex=True, sharey=True)
                    S2Faxs[-1].tick_params(top=False, bottom=False, left=True, right=False, labelbottom=False,labelcolor='black')
                    S2Fwhole.spines['top'].set_color('none')
                    S2Fwhole.spines['bottom'].set_color('none')
                    S2Fwhole.spines['left'].set_color('none')
                    S2Fwhole.spines['right'].set_color('none')
                    S2Fwhole.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                    if m == 'length':
                        S2Fwhole.set_ylabel('%s, %s (px)' % (skel,m),fontsize=16)
                    elif m == 'orientation':
                        S2Fwhole.set_ylabel('%s, %s (°)' % (skel,m), fontsize=16)
                    S2Fwhole.set_xlabel('Belt quadrant', fontsize=16)
                    S2F.suptitle('Belt 1 OFF, Belt 2 ON')

                    #parameters
                    leglabel = ['4cm/s', '8cm/s', '16cm/s', '32cm/s']
                    Pleglabel = ['4cm/s', '8cm/s', '16cm/s', '32cm/s', 'Perception test']
                    Qs = np.arange(4)
                    phase = np.arange(4)
                    width = 0.15
                    nF2S = 0
                    nS2F = 0

                    #plots with quartile along the x-axis
                    # F2Saxs[0].ylabel('Triangle height (px)')
                    # plt.xticks(Qs + width / 2, ('Q1','Q2','Q3','Q4'))
                    for phaseno, phase in enumerate(allPhases[2]):
                        nF2S=0
                        nS2F=0
                        for conIdx, con in enumerate(allConditions):
                            # Fast to Slow
                            if con == 'FastToSlow':
                                try:
                                    F2Saxs[phaseno].bar((conIdx*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[skel,m], width=width,color=colors(nF2S), yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[skel,m]))
                                except:
                                    print('no %s (fast to slow) for file: %s' % (phase, directories[conIdx]))
                                nF2S += 1

                                # Slow to Fast
                            elif con == 'SlowToFast':
                                try:
                                    S2Faxs[phaseno].bar((conIdx*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[skel,m], width=width,color=colors(nS2F), yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[skel,m]))
                                except:
                                    print('no %s (slow to fast) for file: %s' % (phase, directories[conIdx]))
                                nS2F +=1

                            # Perception test
                            elif con == 'PerceptionTest':
                                try:
                                    S2Faxs[phaseno].bar((8*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[skel,m], width=width, hatch="/", color="w", edgecolor="k", yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[skel,m]))
                                except:
                                    print('no %s (perception test) for file: %s' % (phase, directories[conIdx]))


                    F2Saxs[0].legend(labels=leglabel, loc='lower left', bbox_to_anchor=(0,1.01),title='Speed of transition                                                          ', ncol=4, borderaxespad=0, frameon=False, fontsize=9)
                    S2Faxs[0].legend(labels=Pleglabel, loc='lower left', bbox_to_anchor=(-0.05,1.01),title='Speed of transition                                                                                             ', ncol=5, borderaxespad=0, frameon=False, fontsize=9)

                    patchspacing = (F2Saxs[-1].get_xlim()[1] - F2Saxs[-1].get_xlim()[0]) / 4
                    # add bottom boxes for quadrant
                    for num, label in enumerate(['Q1','Q2','Q3','Q4']):
                        F2Saxs[-1].add_patch(plt.Rectangle((F2Saxs[-1].get_xlim()[0] + patchspacing * num, F2Saxs[-1].get_ylim()[0]-(F2Saxs[-1].get_ylim()[1])/5),width=patchspacing, height=(F2Saxs[-1].get_ylim()[1])/5, facecolor=patchcolors(num+1),clip_on=False, linewidth=1, ec='k'))
                        F2Saxs[-1].text((F2Saxs[-1].get_xlim()[0] + patchspacing*(num+1)) - (patchspacing/2), F2Saxs[-1].get_ylim()[0]-(F2Saxs[-1].get_ylim()[1])/7, label, fontsize=15, zorder=5,color='k')
                        F2Saxs[num].spines['right'].set_color('none')
                        S2Faxs[-1].add_patch(plt.Rectangle((S2Faxs[-1].get_xlim()[0] + patchspacing * num, S2Faxs[-1].get_ylim()[0]-(S2Faxs[-1].get_ylim()[1])/5),width=patchspacing, height=(S2Faxs[-1].get_ylim()[1])/5, facecolor=patchcolors(num+1),clip_on=False, linewidth=1, ec='k'))
                        S2Faxs[-1].text((S2Faxs[-1].get_xlim()[0] + patchspacing*(num+1)) - (patchspacing/2), S2Faxs[-1].get_ylim()[0]-(S2Faxs[-1].get_ylim()[1])/7, label, fontsize=15, zorder=5,color='k')
                        S2Faxs[num].spines['right'].set_color('none')
                    F2Saxs[0].spines['right'].set_color('none')
                    F2Saxs[0].spines['top'].set_color('none')
                    S2Faxs[0].spines['right'].set_color('none')
                    S2Faxs[0].spines['top'].set_color('none')
                    # add side boxes for exp phase
                    for num, label in enumerate(['Baseline', 'APA #1', 'APA #2', 'Washout']):
                        F2Saxs[num].add_patch(plt.Rectangle((F2Saxs[num].get_xlim()[1],F2Saxs[num].get_ylim()[0]),width=0.3, height=F2Saxs[num].get_ylim()[1]+abs(F2Saxs[num].get_ylim()[0]), facecolor='w',clip_on=False, linewidth=1, ec='k'))
                        F2Saxs[num].text((F2Saxs[num].get_xlim()[1] + 0.16), (F2Saxs[num].get_ylim()[1]+abs(F2Saxs[num].get_ylim()[0]))/16,  label, fontsize=13, zorder=5, rotation=90, color='k', horizontalalignment='center')
                        S2Faxs[num].add_patch(plt.Rectangle((S2Faxs[num].get_xlim()[1],S2Faxs[num].get_ylim()[0]),width=0.3, height=S2Faxs[num].get_ylim()[1]+abs(S2Faxs[num].get_ylim()[0]), facecolor='w',clip_on=False, linewidth=1, ec='k'))
                        S2Faxs[num].text((S2Faxs[num].get_xlim()[1] + 0.16), (S2Faxs[num].get_ylim()[1]+abs(S2Faxs[num].get_ylim()[0]))/16,  label, fontsize=13, zorder=5, rotation=90, color='k', horizontalalignment='center')


                    F2S.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\FastToSlow\\F2S_%s_%s.png" % (destfolder, skel, m),format='png')
                    S2F.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\SlowToFast\\S2F_%s_%s.png" % (destfolder, skel, m),format='png')


            if m == 'triangle height':
                #parameters
                Qs = np.arange(4)
                phase = np.arange(4)
                width = 0.15
                leglabel = ['4cm/s', '8cm/s', '16cm/s', '32cm/s']
                Pleglabel = ['4cm/s', '8cm/s', '16cm/s', '32cm/s', 'Perception test']

                # format F2S graph
                F2S = plt.figure(figsize=(5.9,6.27)) # to change size of whole figure, when decide on dimensions i want, do F2S.get_figwidth() etc and then adjust accordingly
                F2Swhole = F2S.add_subplot(111)
                F2Sgs = F2S.add_gridspec(4, hspace=0)
                F2Saxs = F2Sgs.subplots(sharex=True, sharey=True)
                F2Saxs[-1].tick_params(top=False, bottom=False, left=True, right=False, labelbottom=False,labelcolor='black')
                F2Swhole.spines['top'].set_color('none')
                F2Swhole.spines['bottom'].set_color('none')
                F2Swhole.spines['left'].set_color('none')
                F2Swhole.spines['right'].set_color('none')
                F2Swhole.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                F2Swhole.set_ylabel('Triangle height (px)', fontsize=16)
                F2Swhole.set_xlabel('Belt quadrant', fontsize=16)
                F2S.suptitle('Belt 1 ON, Belt 2 OFF')

                # format S2F graph
                S2F = plt.figure(figsize=(5.9,6.27))  # to change size of whole figure, when decide on dimensions i want, do F2S.get_figwidth() etc and then adjust accordingly
                S2Fwhole = S2F.add_subplot(111)
                S2Fgs = S2F.add_gridspec(4, hspace=0)
                S2Faxs = S2Fgs.subplots(sharex=True, sharey=True)
                S2Faxs[-1].tick_params(top=False, bottom=False, left=True, right=False, labelbottom=False,labelcolor='black')
                S2Fwhole.spines['top'].set_color('none')
                S2Fwhole.spines['bottom'].set_color('none')
                S2Fwhole.spines['left'].set_color('none')
                S2Fwhole.spines['right'].set_color('none')
                S2Fwhole.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                S2Fwhole.set_ylabel('Triangle height (px)', fontsize=16)
                S2Fwhole.set_xlabel('Belt quadrant', fontsize=16)
                S2F.suptitle('Belt 1 OFF, Belt 2 ON')


                Qs = np.arange(4)
                phase = np.arange(4)
                width = 0.15
                nF2S = 0
                nS2F = 0

                #plots with quartile along the x-axis
                # F2Saxs[0].ylabel('Triangle height (px)')
                # plt.xticks(Qs + width / 2, ('Q1','Q2','Q3','Q4'))
                for phaseno, phase in enumerate(allPhases[2]):
                    nF2S=0
                    nS2F=0
                    for conIdx, con in enumerate(allConditions):
                        # Fast to Slow
                        if con == 'FastToSlow':
                            try:
                                F2Saxs[phaseno].bar((conIdx*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',m], width=width,color=colors(nF2S), yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',m]))
                            except:
                                print('no %s (fast to slow) for file: %s' % (phase, directories[conIdx]))
                            nF2S += 1

                            # Slow to Fast
                        elif con == 'SlowToFast':
                            try:
                                S2Faxs[phaseno].bar((conIdx*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',m], width=width,color=colors(nS2F), yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',m]))
                            except:
                                print('no %s (slow to fast) for file: %s' % (phase, directories[conIdx]))
                            nS2F +=1

                        # Perception test
                        elif con == 'PerceptionTest':
                            try:
                                S2Faxs[phaseno].bar((8*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',m], width=width, hatch="/", color="w", edgecolor="k", yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',m]))
                            except:
                                print('no %s (perception test) for file: %s' % (phase, directories[conIdx]))


                F2Saxs[0].legend(labels=leglabel, loc='lower left', bbox_to_anchor=(0,1.01),title='Speed of transition                                                          ', ncol=4, borderaxespad=0, frameon=False, fontsize=9)
                S2Faxs[0].legend(labels=Pleglabel, loc='lower left', bbox_to_anchor=(-0.1,1.01),title='Speed of transition                                                                                             ', ncol=5, borderaxespad=0, frameon=False, fontsize=9)

                patchspacing = (F2Saxs[-1].get_xlim()[1] - F2Saxs[-1].get_xlim()[0]) / 4
                for num, label in enumerate(['Q1','Q2','Q3','Q4']):
                    F2Saxs[-1].add_patch(plt.Rectangle((F2Saxs[-1].get_xlim()[0] + patchspacing * num, -30),width=patchspacing, height=30, facecolor=patchcolors(num+1),clip_on=False, linewidth=1, ec='k'))
                    F2Saxs[-1].text((F2Saxs[-1].get_xlim()[0] + patchspacing*(num+1)) - (patchspacing/2), -20 , label, fontsize=15, zorder=5,color='k')
                    F2Saxs[num].spines['right'].set_color('none')
                    S2Faxs[-1].add_patch(plt.Rectangle((S2Faxs[-1].get_xlim()[0] + patchspacing * num, -30),width=patchspacing, height=30, facecolor=patchcolors(num+1),clip_on=False, linewidth=1, ec='k'))
                    S2Faxs[-1].text((S2Faxs[-1].get_xlim()[0] + patchspacing*(num+1)) - (patchspacing/2), -20 , label, fontsize=15, zorder=5,color='k')
                    S2Faxs[num].spines['right'].set_color('none')
                F2Saxs[0].spines['right'].set_color('none')
                F2Saxs[0].spines['top'].set_color('none')
                S2Faxs[0].spines['right'].set_color('none')
                S2Faxs[0].spines['top'].set_color('none')
                # add side boxes for exp phase
                for num, label in enumerate(['Baseline', 'APA #1', 'APA #2', 'Washout']):
                    F2Saxs[num].add_patch(plt.Rectangle((F2Saxs[num].get_xlim()[1],0),width=0.3, height=F2Saxs[num].get_ylim()[1], facecolor='w',clip_on=False, linewidth=1, ec='k'))
                    F2Saxs[num].text((F2Saxs[num].get_xlim()[1] + 0.16), 24,  label, fontsize=13, zorder=5, rotation=90, color='k', horizontalalignment='center')
                    S2Faxs[num].add_patch(plt.Rectangle((S2Faxs[num].get_xlim()[1],0),width=0.3, height=S2Faxs[num].get_ylim()[1], facecolor='w',clip_on=False, linewidth=1, ec='k'))
                    S2Faxs[num].text((S2Faxs[num].get_xlim()[1] + 0.16), 24,  label, fontsize=13, zorder=5, rotation=90, color='k', horizontalalignment='center')


                F2S.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\FastToSlow\\F2S_%s.png" % (destfolder, m),
                            format='png')
                S2F.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\SlowToFast\\S2F_%s.png" % (destfolder, m),
                            format='png')

            if m == 'triangle angles':
                angles = ['nose angle', 'hump angle', 'tail1 angle']
                for aIdx, a in enumerate(angles):

                    # format F2S graph
                    F2S = plt.figure(figsize=(5.9,6.27)) # to change size of whole figure, when decide on dimensions i want, do F2S.get_figwidth() etc and then adjust accordingly
                    F2Swhole = F2S.add_subplot(111)
                    F2Sgs = F2S.add_gridspec(4, hspace=0)
                    F2Saxs = F2Sgs.subplots(sharex=True, sharey=True)
                    F2Saxs[-1].tick_params(top=False, bottom=False, left=True, right=False, labelbottom=False,labelcolor='black')
                    F2Swhole.spines['top'].set_color('none')
                    F2Swhole.spines['bottom'].set_color('none')
                    F2Swhole.spines['left'].set_color('none')
                    F2Swhole.spines['right'].set_color('none')
                    F2Swhole.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                    F2Swhole.set_ylabel('%s (°)' % a, fontsize=16)
                    F2Swhole.set_xlabel('Belt quadrant', fontsize=16)
                    F2S.suptitle('Belt 1 ON, Belt 2 OFF')

                    # format S2F graph
                    S2F = plt.figure(figsize=(5.9,6.27))  # to change size of whole figure, when decide on dimensions i want, do F2S.get_figwidth() etc and then adjust accordingly
                    S2Fwhole = S2F.add_subplot(111)
                    S2Fgs = S2F.add_gridspec(4, hspace=0)
                    S2Faxs = S2Fgs.subplots(sharex=True, sharey=True)
                    S2Faxs[-1].tick_params(top=False, bottom=False, left=True, right=False, labelbottom=False,labelcolor='black')
                    S2Fwhole.spines['top'].set_color('none')
                    S2Fwhole.spines['bottom'].set_color('none')
                    S2Fwhole.spines['left'].set_color('none')
                    S2Fwhole.spines['right'].set_color('none')
                    S2Fwhole.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False,labelsize=13)
                    S2Fwhole.set_ylabel('%s (°)' % a, fontsize=16)
                    S2Fwhole.set_xlabel('Belt quadrant', fontsize=16)
                    S2F.suptitle('Belt 1 OFF, Belt 2 ON')

                    #parameters
                    leglabel = ['4cm/s', '8cm/s', '16cm/s', '32cm/s']
                    Pleglabel = ['4cm/s', '8cm/s', '16cm/s', '32cm/s', 'Perception test']
                    Qs = np.arange(4)
                    phase = np.arange(4)
                    width = 0.15
                    nF2S = 0
                    nS2F = 0

                    #plots with quartile along the x-axis
                    # F2Saxs[0].ylabel('Triangle height (px)')
                    # plt.xticks(Qs + width / 2, ('Q1','Q2','Q3','Q4'))
                    for phaseno, phase in enumerate(allPhases[2]):
                        nF2S=0
                        nS2F=0
                        for conIdx, con in enumerate(allConditions):
                            # Fast to Slow
                            if con == 'FastToSlow':
                                try:
                                    F2Saxs[phaseno].bar((conIdx*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',a], width=width,color=colors(nF2S), yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',a]))
                                except:
                                    print('no %s (fast to slow) for file: %s' % (phase, directories[conIdx]))
                                nF2S += 1

                                # Slow to Fast
                            elif con == 'SlowToFast':
                                try:
                                    S2Faxs[phaseno].bar((conIdx*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',a], width=width,color=colors(nS2F), yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',a]))
                                except:
                                    print('no %s (slow to fast) for file: %s' % (phase, directories[conIdx]))
                                nS2F +=1

                            # Perception test
                            elif con == 'PerceptionTest':
                                try:
                                    S2Faxs[phaseno].bar((8*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',a], width=width, hatch="/", color="w", edgecolor="k", yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)['combined',a]))
                                except:
                                    print('no %s (perception test) for file: %s' % (phase, directories[conIdx]))


                    F2Saxs[0].legend(labels=leglabel, loc='lower left', bbox_to_anchor=(0,1.01),title='Speed of transition                                                          ', ncol=4, borderaxespad=0, frameon=False, fontsize=9)
                    S2Faxs[0].legend(labels=Pleglabel, loc='lower left', bbox_to_anchor=(-0.1,1.01),title='Speed of transition                                                                                             ', ncol=5, borderaxespad=0, frameon=False, fontsize=9)

                    patchspacing = (F2Saxs[-1].get_xlim()[1] - F2Saxs[-1].get_xlim()[0]) / 4
                    for num, label in enumerate(['Q1','Q2','Q3','Q4']):
                        F2Saxs[-1].add_patch(plt.Rectangle((F2Saxs[-1].get_xlim()[0] + patchspacing * num, -(F2Saxs[-1].get_ylim()[1])/5),width=patchspacing, height=(F2Saxs[-1].get_ylim()[1])/5, facecolor=patchcolors(num+1),clip_on=False, linewidth=1, ec='k'))
                        F2Saxs[-1].text((F2Saxs[-1].get_xlim()[0] + patchspacing*(num+1)) - (patchspacing/2), -(F2Saxs[-1].get_ylim()[1])/7, label, fontsize=15, zorder=5,color='k')
                        F2Saxs[num].spines['right'].set_color('none')
                        S2Faxs[-1].add_patch(plt.Rectangle((S2Faxs[-1].get_xlim()[0] + patchspacing * num, -(S2Faxs[-1].get_ylim()[1])/5),width=patchspacing, height=(S2Faxs[-1].get_ylim()[1])/5, facecolor=patchcolors(num+1),clip_on=False, linewidth=1, ec='k'))
                        S2Faxs[-1].text((S2Faxs[-1].get_xlim()[0] + patchspacing*(num+1)) - (patchspacing/2), -(S2Faxs[-1].get_ylim()[1])/7, label, fontsize=15, zorder=5,color='k')
                        S2Faxs[num].spines['right'].set_color('none')
                    F2Saxs[0].spines['right'].set_color('none')
                    F2Saxs[0].spines['top'].set_color('none')
                    S2Faxs[0].spines['right'].set_color('none')
                    S2Faxs[0].spines['top'].set_color('none')
                    # add side boxes for exp phase
                    for num, label in enumerate(['Baseline', 'APA #1', 'APA #2', 'Washout']):
                        F2Saxs[num].add_patch(plt.Rectangle((F2Saxs[num].get_xlim()[1],0),width=0.3, height=F2Saxs[num].get_ylim()[1], facecolor='w',clip_on=False, linewidth=1, ec='k'))
                        F2Saxs[num].text((F2Saxs[num].get_xlim()[1] + 0.16), (F2Saxs[num].get_ylim()[1])/8,  label, fontsize=13, zorder=5, rotation=90, color='k', horizontalalignment='center')
                        S2Faxs[num].add_patch(plt.Rectangle((S2Faxs[num].get_xlim()[1],0),width=0.3, height=S2Faxs[num].get_ylim()[1], facecolor='w',clip_on=False, linewidth=1, ec='k'))
                        S2Faxs[num].text((S2Faxs[num].get_xlim()[1] + 0.16), (F2Saxs[num].get_ylim()[1])/8,  label, fontsize=13, zorder=5, rotation=90, color='k', horizontalalignment='center')


                    F2S.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\FastToSlow\\F2S_%s.png" % (destfolder, a),
                                format='png')
                    S2F.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\SlowToFast\\S2F_%s.png" % (destfolder, a),
                                format='png')


            if m == 'TimeSpent (s)' or m == 'Speed (cm/s)':
                times = ['TimeSpent (s)', 'Speed (cm/s)']
                for t in times:
                    # format F2S graph
                    F2S = plt.figure(figsize=(5.9,6.27)) # to change size of whole figure, when decide on dimensions i want, do F2S.get_figwidth() etc and then adjust accordingly
                    F2Swhole = F2S.add_subplot(111)
                    F2Sgs = F2S.add_gridspec(4, hspace=0)
                    F2Saxs = F2Sgs.subplots(sharex=True, sharey=True)
                    F2Saxs[-1].tick_params(top=False, bottom=False, left=True, right=False, labelbottom=False,labelcolor='black')
                    F2Swhole.spines['top'].set_color('none')
                    F2Swhole.spines['bottom'].set_color('none')
                    F2Swhole.spines['left'].set_color('none')
                    F2Swhole.spines['right'].set_color('none')
                    F2Swhole.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                    F2Swhole.set_ylabel(t, fontsize=16)
                    F2Swhole.set_xlabel('Belt quadrant', fontsize=16)
                    F2S.suptitle('Belt 1 ON, Belt 2 OFF')

                    # format S2F graph
                    S2F = plt.figure(figsize=(5.9,6.27))  # to change size of whole figure, when decide on dimensions i want, do F2S.get_figwidth() etc and then adjust accordingly
                    S2Fwhole = S2F.add_subplot(111)
                    S2Fgs = S2F.add_gridspec(4, hspace=0)
                    S2Faxs = S2Fgs.subplots(sharex=True, sharey=True)
                    S2Faxs[-1].tick_params(top=False, bottom=False, left=True, right=False, labelbottom=False,labelcolor='black')
                    S2Fwhole.spines['top'].set_color('none')
                    S2Fwhole.spines['bottom'].set_color('none')
                    S2Fwhole.spines['left'].set_color('none')
                    S2Fwhole.spines['right'].set_color('none')
                    S2Fwhole.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                    S2Fwhole.set_ylabel(t, fontsize=16)
                    S2Fwhole.set_xlabel('Belt quadrant', fontsize=16)
                    S2F.suptitle('Belt 1 OFF, Belt 2 ON')

                    #parameters
                    leglabel = ['4cm/s', '8cm/s', '16cm/s', '32cm/s']
                    Pleglabel = ['4cm/s', '8cm/s', '16cm/s', '32cm/s', 'Perception test']
                    Qs = np.arange(4)
                    phase = np.arange(4)
                    width = 0.15
                    nF2S = 0
                    nS2F = 0

                    #plots with quartile along the x-axis
                    # F2Saxs[0].ylabel('Triangle height (px)')
                    # plt.xticks(Qs + width / 2, ('Q1','Q2','Q3','Q4'))
                    for phaseno, phase in enumerate(allPhases[2]):
                        nF2S=0
                        nS2F=0
                        for conIdx, con in enumerate(allConditions):
                            # Fast to Slow
                            if con == 'FastToSlow':
                                try:
                                    F2Saxs[phaseno].bar((conIdx*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[t], width=width,color=colors(nF2S), yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[t]))
                                except:
                                    print('no %s (fast to slow) for file: %s' % (phase, directories[conIdx]))
                                nF2S += 1

                                # Slow to Fast
                            elif con == 'SlowToFast':
                                try:
                                    S2Faxs[phaseno].bar((conIdx*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[t], width=width,color=colors(nS2F), yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[t]))
                                except:
                                    print('no %s (slow to fast) for file: %s' % (phase, directories[conIdx]))
                                nS2F +=1

                            # Perception test
                            elif con == 'PerceptionTest':
                                try:
                                    S2Faxs[phaseno].bar((8*width)+Qs, allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[t], width=width, hatch="/", color="w", edgecolor="k", yerr=stats.sem(allMeanbyZone[conIdx].loc(axis=0)[phase].loc(axis=1)[t]))
                                except:
                                    print('no %s (perception test) for file: %s' % (phase, directories[conIdx]))


                    F2Saxs[0].legend(labels=leglabel, loc='lower left', bbox_to_anchor=(0,1.01),title='Speed of transition                                                          ', ncol=4, borderaxespad=0, frameon=False, fontsize=9)
                    S2Faxs[0].legend(labels=Pleglabel, loc='lower left', bbox_to_anchor=(-0.05,1.01),title='Speed of transition                                                                                             ', ncol=5, borderaxespad=0, frameon=False, fontsize=9)

                    patchspacing = (F2Saxs[-1].get_xlim()[1] - F2Saxs[-1].get_xlim()[0]) / 4
                    # add bottom boxes for quadrant
                    for num, label in enumerate(['Q1','Q2','Q3','Q4']):
                        F2Saxs[-1].add_patch(plt.Rectangle((F2Saxs[-1].get_xlim()[0] + patchspacing * num, F2Saxs[-1].get_ylim()[0]-(F2Saxs[-1].get_ylim()[1])/5),width=patchspacing, height=(F2Saxs[-1].get_ylim()[1])/5, facecolor=patchcolors(num+1),clip_on=False, linewidth=1, ec='k'))
                        F2Saxs[-1].text((F2Saxs[-1].get_xlim()[0] + patchspacing*(num+1)) - (patchspacing/2), F2Saxs[-1].get_ylim()[0]-(F2Saxs[-1].get_ylim()[1])/7, label, fontsize=15, zorder=5,color='k')
                        F2Saxs[num].spines['right'].set_color('none')
                        S2Faxs[-1].add_patch(plt.Rectangle((S2Faxs[-1].get_xlim()[0] + patchspacing * num, S2Faxs[-1].get_ylim()[0]-(S2Faxs[-1].get_ylim()[1])/5),width=patchspacing, height=(S2Faxs[-1].get_ylim()[1])/5, facecolor=patchcolors(num+1),clip_on=False, linewidth=1, ec='k'))
                        S2Faxs[-1].text((S2Faxs[-1].get_xlim()[0] + patchspacing*(num+1)) - (patchspacing/2), S2Faxs[-1].get_ylim()[0]-(S2Faxs[-1].get_ylim()[1])/7, label, fontsize=15, zorder=5,color='k')
                        S2Faxs[num].spines['right'].set_color('none')
                    F2Saxs[0].spines['right'].set_color('none')
                    F2Saxs[0].spines['top'].set_color('none')
                    S2Faxs[0].spines['right'].set_color('none')
                    S2Faxs[0].spines['top'].set_color('none')
                    # add side boxes for exp phase
                    for num, label in enumerate(['Baseline', 'APA #1', 'APA #2', 'Washout']):
                        F2Saxs[num].add_patch(plt.Rectangle((F2Saxs[num].get_xlim()[1],F2Saxs[num].get_ylim()[0]),width=0.3, height=F2Saxs[num].get_ylim()[1]+abs(F2Saxs[num].get_ylim()[0]), facecolor='w',clip_on=False, linewidth=1, ec='k'))
                        F2Saxs[num].text((F2Saxs[num].get_xlim()[1] + 0.16), (F2Saxs[num].get_ylim()[1]+abs(F2Saxs[num].get_ylim()[0]))/16,  label, fontsize=13, zorder=5, rotation=90, color='k', horizontalalignment='center')
                        S2Faxs[num].add_patch(plt.Rectangle((S2Faxs[num].get_xlim()[1],S2Faxs[num].get_ylim()[0]),width=0.3, height=S2Faxs[num].get_ylim()[1]+abs(S2Faxs[num].get_ylim()[0]), facecolor='w',clip_on=False, linewidth=1, ec='k'))
                        S2Faxs[num].text((S2Faxs[num].get_xlim()[1] + 0.16), (S2Faxs[num].get_ylim()[1]+abs(S2Faxs[num].get_ylim()[0]))/16,  label, fontsize=13, zorder=5, rotation=90, color='k', horizontalalignment='center')

                    if t == 'Speed (cm/s)':
                        F2S.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\FastToSlow\\F2S_Speed.png" % destfolder,
                                    format='png')
                        S2F.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\SlowToFast\\S2F_Speed.png" % destfolder,
                                    format='png')
                    else:
                        F2S.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\FastToSlow\\F2S_%s.png" % (destfolder, t),
                                    format='png')
                        S2F.savefig("%s\\plots\\APA_Char\\byZone\\20210607\\SlowToFast\\S2F_%s.png" % (destfolder, t),
                                    format='png')





    def plotAPAChar(self, directories, destfolder, error, colormap='cool'):
        ################## put in skel, measure and 'fast_slow' or 'slow_fast' as a function input #####################
        ################## each graph should have a line for each phase and each degree of the speed transition ##########
        ###### input should be a list of the directories with APAChar files in them #########

        allMeanbyTime = list()
        allMeanbyRuns = list()
        allStdbyTime = list()
        allStdbyRuns = list()
        allSembyTime = list()
        allSembyRuns = list()
        allConditions = list()
        allSpeeds = list()

        skeletonList = ['Tail2_Tail3', 'Nose_Hump', 'Shoulder_Hump', 'Hip_RAnkle', 'Nose_Tail1', 'Hip_Tail1',
                        'Tail1_Tail2', 'Shoulder_Tail1', 'Hump_RAnkle', 'RHindpaw_RAnkle', 'Hump_Hip',
                        'Nose_Shoulder', 'Hump_RForepaw', 'Tail1_Tail3', 'Shoulder_RForepaw', 'Hump_Tail1']
        measure = ['length', 'orientation', 'triangle height', 'triangle angles']
        runs = list(range(1, 31))

        for dir in range(0, len(directories)):
            meanbyTime = pd.read_hdf(
                "%s\\%s_SkelMeanGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
            meanbyRuns = pd.read_hdf(
                "%s\\%s_SkelMeanGroup_byRuns.h5" % (directories[dir], os.path.basename(directories[dir])))
            stdbyTime = pd.read_hdf(
                "%s\\%s_SkelStdGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
            stdbyRuns = pd.read_hdf(
                "%s\\%s_SkelStdGroup_byRuns.h5" % (directories[dir], os.path.basename(directories[dir])))
            sembyTime = pd.read_hdf(
                "%s\\%s_SkelSemGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
            sembyRuns = pd.read_hdf(
                "%s\\%s_SkelSemGroup_byRuns.h5" % (directories[dir], os.path.basename(directories[dir])))

            # Check which day of APA Characterise experiments data is from
            ############################################################################################################
            ##################### THIS CAN NOW BE GOT FROM GET_EXP_DETAILS FUNCTION IN UTILS.PY ########################
            ############################################################################################################
            exp = Utils().get_exp_details(directories[dir])
            condition = exp['condition']
            speed = exp['Acspeed']
            perceivedspeed = exp['Pdspeed']

            allMeanbyTime.append(meanbyTime)
            allMeanbyRuns.append(meanbyRuns)
            allStdbyTime.append(stdbyTime)
            allStdbyRuns.append(stdbyRuns)
            allSembyTime.append(sembyRuns)
            allSembyRuns.append(sembyRuns)
            allConditions.append(condition)
            allSpeeds.append(speed)

        if error == 'std':
            allStdbyRuns = allStdbyRuns
        elif error == 'sem':
            allStdbyRuns = allSembyRuns

        ## plot 'byTime' data, ie plot line graphs of each measure and each skeleton for baseline, APA and washout
        #################################

        ## plot 'byRuns' data, ie on each plot there is a line graph with runs on x axis and all measures (seperately) on y axis. Do a seperate graph for each measure, skeleton and speed direction. Plot multiple speeds (of same direction) on a single plot.
        colors = Utils().get_cmap(n=4, name=colormap)
        for m in measure:
            if m == 'length' or m == 'orientation':
                for skelIdx, skel in enumerate(skeletonList):
                    if skel == 'Nose_Tail1':
                        # set parameters for fast to slow fig
                        plt.figure(num="%s, %s, Fast to Slow" % (m, skel), figsize=(11, 4))
                        axF2S = plt.subplot(111)
                        axF2S.set_xlim(0, max(runs))
                        if m == 'orientation':
                            axF2S.set_ylim(0, 300)
                            patchwidth = 40
                        if m == 'length':
                            axF2S.set_ylim(0, 650)
                            patchwidth = 70
                        axF2S.tick_params(axis='y', labelsize=12)
                        plt.tick_params(
                            axis='x',  # changes apply to the x-axis
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            labelbottom=False)
                        axF2S.add_patch(
                            plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                                          linewidth=0))
                        axF2S.add_patch(
                            plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                                          linewidth=0))
                        axF2S.add_patch(
                            plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                                          linewidth=0))
                        axF2S.text(1.2, -patchwidth + patchwidth * 0.1, ' Baseline\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')
                        axF2S.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=15, zorder=5,
                                   color='k')
                        axF2S.text(26.2, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')

                        # set params for slow to fast fig
                        plt.figure(num="%s, %s, Slow to Fast" % (m, skel), figsize=(11, 4))
                        axS2F = plt.subplot(111)
                        axS2F.set_xlim(0, max(runs))
                        if m == 'orientation':
                            axS2F.set_ylim(0, 300)
                            patchwidth = 40
                        if m == 'length':
                            axS2F.set_ylim(0, 600)
                            patchwidth = 70
                        axS2F.tick_params(axis='y', labelsize=12)
                        plt.tick_params(
                            axis='x',  # changes apply to the x-axis
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            labelbottom=False)
                        axS2F.add_patch(
                            plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                                          linewidth=0, ))
                        axS2F.add_patch(
                            plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                                          linewidth=0))
                        axS2F.add_patch(
                            plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                                          linewidth=0))
                        axS2F.text(1.2, -patchwidth + patchwidth * 0.1, 'Baseline\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')
                        axS2F.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=15, zorder=5,
                                   color='k')
                        axS2F.text(26.2, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')

                        # plt.figure(num="%s, %s, Perception Test" % (m, skel), figsize=(11, 4))
                        # axP = plt.subplot(111)
                        # axP.set_xlim(0, max(runs))
                        # if m == 'orientation':
                        #     axS2F.set_ylim(0, 300)
                        #     patchwidth = 40
                        # if m == 'length':
                        #     axS2F.set_ylim(0, 600)
                        #     patchwidth = 70
                        # plt.tick_params(
                        #     axis='x',  # changes apply to the x-axis
                        #     which='both',  # both major and minor ticks are affected
                        #     bottom=False,  # ticks along the bottom edge are off
                        #     top=False,  # ticks along the top edge are off
                        #     labelbottom=False)
                        # axP.add_patch(
                        #     plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                        #                   linewidth=0, ))
                        # axP.add_patch(
                        #     plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                        #                   linewidth=0))
                        # axP.add_patch(
                        #     plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                        #                   linewidth=0))
                        # axP.text(1.5, -patchwidth + patchwidth * 0.1, ' Baseline\nTrials = 5', fontsize=10, zorder=5,
                        #          color='k')
                        # axP.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=10, zorder=5,
                        #          color='k')
                        # axP.text(26.7, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=10, zorder=5,
                        #          color='k')

                        # ax.set_ylim(0, 250)
                        nF2S = 0
                        nS2F = 0
                        for conIdx, con in enumerate(allConditions):

                            # Fast to Slow
                            if con == 'FastToSlow':
                                plt.figure(num="%s, %s, Fast to Slow" % (m, skel))
                                plt.title('Fast to Slow Transition - %s - %s' % (skel, m))
                                axF2S.plot(allMeanbyRuns[conIdx].loc(axis=1)[skel, m].index.values + 1,
                                           allMeanbyRuns[conIdx].loc(axis=1)[skel, m], label=allSpeeds[conIdx],
                                           color=colors(nF2S))
                                axF2S.fill_between(allMeanbyRuns[conIdx].loc(axis=1)[skel, m].index.values + 1,
                                                   allMeanbyRuns[conIdx].loc(axis=1)[skel, m] -
                                                   allStdbyRuns[conIdx].loc(axis=1)[skel, m],
                                                   allMeanbyRuns[conIdx].loc(axis=1)[skel, m] +
                                                   allStdbyRuns[conIdx].loc(axis=1)[skel, m],
                                                   interpolate=False, alpha=0.1, color=colors(nF2S))

                                nF2S += 1  # MUST BE LAST

                            elif con == 'SlowToFast':
                                plt.figure(num="%s, %s, Slow to Fast" % (m, skel))
                                plt.title('Slow to Fast Transition - %s - %s' % (skel, m))
                                plt.plot(allMeanbyRuns[conIdx].loc(axis=1)[skel, m].index.values + 1,
                                         allMeanbyRuns[conIdx].loc(axis=1)[skel, m], label=allSpeeds[conIdx],
                                         color=colors(nS2F))
                                plt.fill_between(allMeanbyRuns[conIdx].loc(axis=1)[skel, m].index.values + 1,
                                                 allMeanbyRuns[conIdx].loc(axis=1)[skel, m] -
                                                 allStdbyRuns[conIdx].loc(axis=1)[skel, m],
                                                 allMeanbyRuns[conIdx].loc(axis=1)[skel, m] +
                                                 allStdbyRuns[conIdx].loc(axis=1)[skel, m],
                                                 interpolate=False, alpha=0.1, color=colors(nS2F))

                                nS2F += 1  # MUST BE LAST

                            # elif con == 'PerceptionTest':
                            #     plt.figure(num="%s, %s, Perception Test" % (m, skel))
                            #     plt.title('Perception Test - %s - %s' % (m, skel))
                            #     plt.plot(allMeanbyRuns[conIdx].loc(axis=1)[skel, m].index.values + 1,
                            #              allMeanbyRuns[conIdx].loc(axis=1)[skel, m], label=allSpeeds[conIdx],
                            #              color=colors(0))
                            #     plt.fill_between(allMeanbyRuns[conIdx].loc(axis=1)[skel, m].index.values + 1,
                            #                      allMeanbyRuns[conIdx].loc(axis=1)[skel, m] -
                            #                      allStdbyRuns[conIdx].loc(axis=1)[skel, m],
                            #                      allMeanbyRuns[conIdx].loc(axis=1)[skel, m] +
                            #                      allStdbyRuns[conIdx].loc(axis=1)[skel, m],
                            #                      interpolate=False, alpha=0.1, color=colors(0))
                            #     plt.figure(num="%s, %s, Perception Test" % (m, skel))
                            #     plt.plot(allMeanbyRuns[6].loc(axis=1)[skel, m].index.values + 1,
                            #              allMeanbyRuns[6].loc(axis=1)[skel, m], label=allSpeeds[6],
                            #              color=colors(2))
                            #     plt.fill_between(allMeanbyRuns[6].loc(axis=1)[skel, m].index.values + 1,
                            #                      allMeanbyRuns[6].loc(axis=1)[skel, m] -
                            #                      allStdbyRuns[6].loc(axis=1)[skel, m],
                            #                      allMeanbyRuns[6].loc(axis=1)[skel, m] +
                            #                      allStdbyRuns[6].loc(axis=1)[skel, m],
                            #                      interpolate=False, alpha=0.1, color=colors(2))

                        if error == 'std':
                            plt.figure(num="%s, %s, Fast to Slow" % (m, skel))
                            #plt.legend(title='Speed of transition (m/s)')
                            plt.savefig("%s\\plots\\APA_Char\\byRuns\\FastToSlow\\F2S_%s_%s.png" % (destfolder, skel, m),
                                        format='png')

                            plt.figure(num="%s, %s, Slow to Fast" % (m, skel))
                            #plt.legend(title='Speed of transition (m/s)')
                            plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\S2F_%s_%s.png" % (destfolder, skel, m),
                                        format='png')

                            # plt.figure(num="%s, %s, Perception Test" % (m, skel))
                            # plt.legend(title='Speed of transition (m/s)')
                            # plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\PerceptionTest_%s_%s.png" % (destfolder, skel, m),format='png')

                        elif error == 'sem':
                            plt.figure(num="%s, %s, Fast to Slow" % (m, skel))
                            #plt.legend(title='Speed of transition (m/s)')
                            plt.savefig("%s\\plots\\APA_Char\\byRuns\\FastToSlow\\F2S_%s_%s_SEM.png" % (destfolder, skel, m),
                                        format='png')

                            plt.figure(num="%s, %s, Slow to Fast" % (m, skel))
                            #plt.legend(title='Speed of transition (m/s)')
                            plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\S2F_%s_%s_SEM.png" % (destfolder, skel, m),
                                        format='png')

                            # plt.figure(num="%s, %s, Perception Test" % (m, skel))
                            # plt.legend(title='Speed of transition (m/s)')
                            # plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\PerceptionTest_%s_%s_SEM.png" % (destfolder, skel, m),format='png')


            if m == 'triangle height':
                patchwidth = 30
                plt.figure(num="%s, Fast to Slow" % m, figsize=(11, 4))
                axF2S = plt.subplot(111)
                axF2S.set_xlim(0, max(runs))
                axF2S.set_ylim(0, 200)
                axF2S.tick_params(axis='y', labelsize=12)
                plt.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)
                axF2S.add_patch(plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                                              linewidth=0))
                axF2S.add_patch(
                    plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                                  linewidth=0))
                axF2S.add_patch(
                    plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                                  linewidth=0))
                axF2S.text(1.2, -patchwidth + patchwidth * 0.1, 'Baseline\nTrials = 5', fontsize=15, zorder=5,
                           color='k')
                axF2S.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=15, zorder=5,
                           color='k')
                axF2S.text(26.2, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=15, zorder=5,
                           color='k')

                plt.figure(num="%s, Slow to Fast" % m, figsize=(11, 4))
                axS2F = plt.subplot(111)
                axS2F.set_xlim(0, max(runs))
                axS2F.set_ylim(0, 200)
                axS2F.tick_params(axis='y', labelsize=12)
                plt.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)
                axS2F.add_patch(plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                                              linewidth=0, ))
                axS2F.add_patch(
                    plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                                  linewidth=0))
                axS2F.add_patch(
                    plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                                  linewidth=0))
                axS2F.text(1.2, -patchwidth + patchwidth * 0.1, 'Baseline\nTrials = 5', fontsize=15, zorder=5,
                           color='k')
                axS2F.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=15, zorder=5,
                           color='k')
                axS2F.text(26.2, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=15, zorder=5,
                           color='k')

                # plt.figure(num="%s, Perception Test" % m, figsize=(11, 4))
                # axP = plt.subplot(111)
                # axP.set_xlim(0, max(runs))
                # axP.set_ylim(0, 200)
                # plt.tick_params(
                #     axis='x',  # changes apply to the x-axis
                #     which='both',  # both major and minor ticks are affected
                #     bottom=False,  # ticks along the bottom edge are off
                #     top=False,  # ticks along the top edge are off
                #     labelbottom=False)
                # axP.add_patch(plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                #                             linewidth=0, ))
                # axP.add_patch(
                #     plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                #                   linewidth=0))
                # axP.add_patch(plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                #                             linewidth=0))
                # axP.text(1.5, -patchwidth + patchwidth * 0.1, ' Baseline\nTrials = 5', fontsize=10, zorder=5,
                #          color='k')
                # axP.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=10, zorder=5,
                #          color='k')
                # axP.text(26.7, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=10, zorder=5,
                #          color='k')
                # ax.set_ylim(0, 250)
                nF2S = 0
                nS2F = 0

                for conIdx, con in enumerate(allConditions):

                    # Fast to Slow
                    if con == 'FastToSlow':
                        plt.figure(num="%s, Fast to Slow" % m)
                        plt.title('Fast to Slow Transition - %s' % m)
                        axF2S.plot(allMeanbyRuns[conIdx].loc(axis=1)['combined', m].index.values + 1,
                                   allMeanbyRuns[conIdx].loc(axis=1)['combined', m], label=allSpeeds[conIdx],
                                   color=colors(nF2S))
                        axF2S.fill_between(allMeanbyRuns[conIdx].loc(axis=1)['combined', m].index.values + 1,
                                           allMeanbyRuns[conIdx].loc(axis=1)['combined', m] -
                                           allStdbyRuns[conIdx].loc(axis=1)['combined', m],
                                           allMeanbyRuns[conIdx].loc(axis=1)['combined', m] +
                                           allStdbyRuns[conIdx].loc(axis=1)['combined', m],
                                           interpolate=False, alpha=0.1, color=colors(nF2S))

                        nF2S += 1  # MUST BE LAST

                    elif con == 'SlowToFast':
                        plt.figure(num="%s, Slow to Fast" % m)
                        plt.title('Slow to Fast Transition - %s' % m)
                        plt.plot(allMeanbyRuns[conIdx].loc(axis=1)['combined', m].index.values + 1,
                                 allMeanbyRuns[conIdx].loc(axis=1)['combined', m], label=allSpeeds[conIdx],
                                 color=colors(nS2F))
                        plt.fill_between(allMeanbyRuns[conIdx].loc(axis=1)['combined', m].index.values + 1,
                                         allMeanbyRuns[conIdx].loc(axis=1)['combined', m] -
                                         allStdbyRuns[conIdx].loc(axis=1)['combined', m],
                                         allMeanbyRuns[conIdx].loc(axis=1)['combined', m] +
                                         allStdbyRuns[conIdx].loc(axis=1)['combined', m],
                                         interpolate=False, alpha=0.1, color=colors(nS2F))
                        nS2F += 1  # MUST BE LAST

                    # elif con == 'PerceptionTest':
                    #     plt.figure(num="%s, Perception Test" % m)
                    #     plt.title('Perception Test - %s' % m)
                    #     plt.plot(allMeanbyRuns[conIdx].loc(axis=1)['combined', m].index.values + 1,
                    #              allMeanbyRuns[conIdx].loc(axis=1)['combined', m], label=allSpeeds[conIdx],
                    #              color=colors(0))
                    #     plt.fill_between(allMeanbyRuns[conIdx].loc(axis=1)['combined', m].index.values + 1,
                    #                      allMeanbyRuns[conIdx].loc(axis=1)['combined', m] -
                    #                      allStdbyRuns[conIdx].loc(axis=1)['combined', m],
                    #                      allMeanbyRuns[conIdx].loc(axis=1)['combined', m] +
                    #                      allStdbyRuns[conIdx].loc(axis=1)['combined', m],
                    #                      interpolate=False, alpha=0.1, color=colors(0))
                    #     plt.figure(num="%s, Perception Test" % m)
                    #     plt.plot(allMeanbyRuns[6].loc(axis=1)['combined', m].index.values + 1,
                    #              allMeanbyRuns[6].loc(axis=1)['combined', m], label=allSpeeds[6],
                    #              color=colors(2))
                    #     plt.fill_between(allMeanbyRuns[6].loc(axis=1)['combined', m].index.values + 1,
                    #                      allMeanbyRuns[6].loc(axis=1)['combined', m] -
                    #                      allStdbyRuns[6].loc(axis=1)['combined', m],
                    #                      allMeanbyRuns[6].loc(axis=1)['combined', m] +
                    #                      allStdbyRuns[6].loc(axis=1)['combined', m],
                    #                      interpolate=False, alpha=0.1, color=colors(2))

                if error == 'std':
                    plt.figure(num="%s, Fast to Slow" % m)
                    #plt.legend(title='Speed of transition (m/s)')
                    plt.savefig("%s\\plots\\APA_Char\\byRuns\\FastToSlow\\F2S_%s.png" % (destfolder, m),
                                format='png')

                    plt.figure(num="%s, Slow to Fast" % m)
                    #plt.legend(title='Speed of transition (m/s)')
                    plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\S2F_%s.png" % (destfolder, m),
                                format='png')

                    plt.figure(num="%s, Perception Test" % m)
                    #plt.legend(title='Speed of transition (m/s)')
                    plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\PerceptionTest_%s.png" % (destfolder, m),format='png')

                elif error == 'sem':
                    plt.figure(num="%s, Fast to Slow" % m)
                    #plt.legend(title='Speed of transition (m/s)')
                    plt.savefig("%s\\plots\\APA_Char\\byRuns\\FastToSlow\\F2S_%s_SEM.png" % (destfolder, m),
                                format='png')

                    plt.figure(num="%s, Slow to Fast" % m)
                    #plt.legend(title='Speed of transition (m/s)')
                    plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\S2F_%s_SEM.png" % (destfolder, m),
                                format='png')

                    plt.figure(num="%s, Perception Test" % m)
                   # plt.legend(title='Speed of transition (m/s)')
                    plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\PerceptionTest_%s_SEM.png" % (destfolder, m),format='png')

            if m == 'triangle angles':
                angles = ['nose angle', 'hump angle', 'tail1 angle']
                for aIdx, a in enumerate(angles):
                    plt.figure(num="%s, Fast to Slow" % a, figsize=(11, 4))
                    axF2S = plt.subplot(111)
                    axF2S.set_xlim(0, max(runs))
                    if a == 'hump angle':
                        axF2S.set_ylim(0, 175)
                        patchwidth = 20
                    else:
                        axF2S.set_ylim(0, 80)
                        patchwidth = 10
                    axF2S.tick_params(axis='y', labelsize=12)
                    plt.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)
                    axF2S.add_patch(
                        plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                                      linewidth=0))
                    axF2S.add_patch(
                        plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                                      linewidth=0))
                    axF2S.add_patch(
                        plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                                      linewidth=0))
                    axF2S.text(1.2, -patchwidth + patchwidth * 0.1, 'Baseline\nTrials = 5', fontsize=15, zorder=5,
                               color='k')
                    axF2S.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=15, zorder=5,
                               color='k')
                    axF2S.text(26.2, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=15, zorder=5,
                               color='k')

                    plt.figure(num="%s, Slow to Fast" % a, figsize=(11, 4))
                    axS2F = plt.subplot(111)
                    axS2F.set_xlim(0, max(runs))
                    if a == 'hump angle':
                        axS2F.set_ylim(0, 175)
                        patchwidth = 20
                    else:
                        axS2F.set_ylim(0, 80)
                        patchwidth = 10
                    axS2F.tick_params(axis='y', labelsize=12)
                    plt.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)
                    axS2F.add_patch(
                        plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                                      linewidth=0, ))
                    axS2F.add_patch(
                        plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                                      linewidth=0))
                    axS2F.add_patch(
                        plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                                      linewidth=0))
                    axS2F.text(1.2, -patchwidth + patchwidth * 0.1, 'Baseline\nTrials = 5', fontsize=15, zorder=5,
                               color='k')
                    axS2F.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=15, zorder=5,
                               color='k')
                    axS2F.text(26.2, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=15, zorder=5,
                               color='k')

                    # plt.figure(num="%s, Perception Test" % m, figsize=(11, 4))
                    # axP = plt.subplot(111)
                    # axP.set_xlim(0, max(runs))
                    # if a == 'hump angle':
                    #     axS2F.set_ylim(0, 175)
                    #     patchwidth = 20
                    # else:
                    #     axS2F.set_ylim(0, 80)
                    #     patchwidth = 10
                    # plt.tick_params(
                    #     axis='x',  # changes apply to the x-axis
                    #     which='both',  # both major and minor ticks are affected
                    #     bottom=False,  # ticks along the bottom edge are off
                    #     top=False,  # ticks along the top edge are off
                    #     labelbottom=False)
                    # axP.add_patch(
                    #     plt.Rectangle((0, -patchwidth), 5, patchwidth, facecolor='lightblue', clip_on=False,
                    #                   linewidth=0, ))
                    # axP.add_patch(
                    #     plt.Rectangle((5, -patchwidth), 20, patchwidth, facecolor='lightskyblue', clip_on=False,
                    #                   linewidth=0))
                    # axP.add_patch(
                    #     plt.Rectangle((25, -patchwidth), 5, patchwidth, facecolor='dodgerblue', clip_on=False,
                    #                   linewidth=0))
                    # axP.text(1.5, -patchwidth + patchwidth * 0.1, ' Baseline\nTrials = 5', fontsize=10, zorder=5,
                    #          color='k')
                    # axP.text(13.5, -patchwidth + patchwidth * 0.1, '     APA\nTrials = 20', fontsize=10, zorder=5,
                    #          color='k')
                    # axP.text(26.7, -patchwidth + patchwidth * 0.1, 'Washout\nTrials = 5', fontsize=10, zorder=5,
                    #          color='k')

                    # ax.set_ylim(0, 250)
                    nF2S = 0
                    nS2F = 0

                    for conIdx, con in enumerate(allConditions):

                        # Fast to Slow
                        if con == 'FastToSlow':
                            plt.figure(num="%s, Fast to Slow" % a)
                            plt.title('Fast to Slow Transition - %s' % a)
                            axF2S.plot(allMeanbyRuns[conIdx].loc(axis=1)['combined', a].index.values + 1,
                                       allMeanbyRuns[conIdx].loc(axis=1)['combined', a], label=allSpeeds[conIdx],
                                       color=colors(nF2S))
                            axF2S.fill_between(allMeanbyRuns[conIdx].loc(axis=1)['combined', a].index.values + 1,
                                               allMeanbyRuns[conIdx].loc(axis=1)['combined', a] -
                                               allStdbyRuns[conIdx].loc(axis=1)['combined', a],
                                               allMeanbyRuns[conIdx].loc(axis=1)['combined', a] +
                                               allStdbyRuns[conIdx].loc(axis=1)['combined', a],
                                               interpolate=False, alpha=0.1, color=colors(nF2S))

                            nF2S += 1  # MUST BE LAST

                        elif con == 'SlowToFast':
                            plt.figure(num="%s, Slow to Fast" % a)
                            plt.title('Slow to Fast Transition - %s' % a)
                            plt.plot(allMeanbyRuns[conIdx].loc(axis=1)['combined', a].index.values + 1,
                                     allMeanbyRuns[conIdx].loc(axis=1)['combined', a], label=allSpeeds[conIdx],
                                     color=colors(nS2F))
                            plt.fill_between(allMeanbyRuns[conIdx].loc(axis=1)['combined', a].index.values + 1,
                                             allMeanbyRuns[conIdx].loc(axis=1)['combined', a] -
                                             allStdbyRuns[conIdx].loc(axis=1)['combined', a],
                                             allMeanbyRuns[conIdx].loc(axis=1)['combined', a] +
                                             allStdbyRuns[conIdx].loc(axis=1)['combined', a],
                                             interpolate=False, alpha=0.1, color=colors(nS2F))

                            nS2F += 1  # MUST BE LAST

                        # elif con == 'PerceptionTest':
                        #     plt.figure(num="%s, Perception Test" % a)
                        #     plt.title('Perception Test - %s' % a)
                        #     plt.plot(allMeanbyRuns[conIdx].loc(axis=1)['combined', a].index.values + 1,
                        #              allMeanbyRuns[conIdx].loc(axis=1)['combined', a], label=allSpeeds[conIdx],
                        #              color=colors(0))
                        #     plt.fill_between(allMeanbyRuns[conIdx].loc(axis=1)['combined', a].index.values + 1,
                        #                      allMeanbyRuns[conIdx].loc(axis=1)['combined', a] -
                        #                      allStdbyRuns[conIdx].loc(axis=1)['combined', a],
                        #                      allMeanbyRuns[conIdx].loc(axis=1)['combined', a] +
                        #                      allStdbyRuns[conIdx].loc(axis=1)['combined', a],
                        #                      interpolate=False, alpha=0.1, color=colors(0))
                        #     plt.figure(num="%s, Perception Test" % a)
                        #     plt.plot(allMeanbyRuns[6].loc(axis=1)['combined', a].index.values + 1,
                        #              allMeanbyRuns[6].loc(axis=1)['combined', a], label=allSpeeds[6],
                        #              color=colors(2))
                        #     plt.fill_between(allMeanbyRuns[6].loc(axis=1)['combined', a].index.values + 1,
                        #                      allMeanbyRuns[6].loc(axis=1)['combined', a] -
                        #                      allStdbyRuns[6].loc(axis=1)['combined', a],
                        #                      allMeanbyRuns[6].loc(axis=1)['combined', a] +
                        #                      allStdbyRuns[6].loc(axis=1)['combined', a],
                        #                      interpolate=False, alpha=0.1, color=colors(2))

                    if error == 'std':
                        plt.figure(num="%s, Fast to Slow" % a)
                        #plt.legend(title='Speed of transition (m/s)')
                        plt.savefig("%s\\plots\\APA_Char\\byRuns\\FastToSlow\\F2S_%s.png" % (destfolder, a),
                                    format='png')

                        plt.figure(num="%s, Slow to Fast" % a)
                        #plt.legend(title='Speed of transition (m/s)')
                        plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\S2F_%s.png" % (destfolder, a),
                                    format='png')

                        # plt.figure(num="%s, Perception Test" % a)
                        # plt.legend(title='Speed of transition (m/s)')
                        # plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\PerceptionTest_%s.png" % (destfolder, a),format='png')
                    elif error == 'sem':
                        plt.figure(num="%s, Fast to Slow" % a)
                        #plt.legend(title='Speed of transition (m/s)')
                        plt.savefig("%s\\plots\\APA_Char\\byRuns\\FastToSlow\\F2S_%s_SEM.png" % (destfolder, a),
                                    format='png')

                        plt.figure(num="%s, Slow to Fast" % a)
                        #plt.legend(title='Speed of transition (m/s)')
                        plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\S2F_%s_SEM.png" % (destfolder, a),
                                    format='png')

                        # plt.figure(num="%s, Perception Test" % a)
                        # plt.legend(title='Speed of transition (m/s)')
                        # plt.savefig("%s\\plots\\APA_Char\\byRuns\\SlowToFast\\PerceptionTest_%s_SEM.png" % (destfolder, a),format='png')


    def plotAPAVMT(self, directories, destfolder, error, webdir='no', colormap='cool'):
        ################## put in skel, measure and 'fast_slow' or 'slow_fast' as a function input #####################
        ################## each graph should have a line for each phase and each degree of the speed transition ##########
        ###### input should be a list of the directories with APAChar files in them #########
        if webdir == 'yes':
            directories = [r"M:\DLC_DualBelt_webcam-Holly-2021-04-22\analysed_data\iteration-2\20201214",
                           r"M:\DLC_DualBelt_webcam-Holly-2021-04-22\analysed_data\iteration-2\20201216",
                           r"M:\DLC_DualBelt_webcam-Holly-2021-04-22\analysed_data\iteration-2\20201217",
                           r"M:\DLC_DualBelt_webcam-Holly-2021-04-22\analysed_data\iteration-2\20201218"
                            ]

        allMeanbyTime = list()
        allMeanbyRuns = list()
        allStdbyTime = list()
        allStdbyRuns = list()
        allSembyTime = list()
        allSembyRuns = list()
        allConditions = list()
        allSpeeds = list()
        allVMTcons = list()
        allVMTtypes = list()
        alllabels = list()
        allMeanWait = list()
        allSemWait =list()

        skeletonList = ['Tail2_Tail3', 'Nose_Hump', 'Shoulder_Hump', 'Hip_RAnkle', 'Nose_Tail1', 'Hip_Tail1',
                        'Tail1_Tail2', 'Shoulder_Tail1', 'Hump_RAnkle', 'RHindpaw_RAnkle', 'Hump_Hip',
                        'Nose_Shoulder', 'Hump_RForepaw', 'Tail1_Tail3', 'Shoulder_RForepaw', 'Hump_Tail1']
        measure = ['length', 'orientation', 'triangle height', 'triangle angles']
        runs = list(range(1, 36))

        for dir in range(0, len(directories)):
            if webdir == 'no':
                meanbyTime = pd.read_hdf(
                    "%s\\%s_SkelMeanGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
                meanbyRuns = pd.read_hdf(
                    "%s\\%s_SkelMeanGroup_byRuns.h5" % (directories[dir], os.path.basename(directories[dir])))
                stdbyTime = pd.read_hdf(
                    "%s\\%s_SkelStdGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
                stdbyRuns = pd.read_hdf(
                    "%s\\%s_SkelStdGroup_byRuns.h5" % (directories[dir], os.path.basename(directories[dir])))
                sembyTime = pd.read_hdf(
                    "%s\\%s_SkelSemGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
                sembyRuns = pd.read_hdf(
                    "%s\\%s_SkelSemGroup_byRuns.h5" % (directories[dir], os.path.basename(directories[dir])))

                allMeanbyTime.append(meanbyTime)
                allMeanbyRuns.append(meanbyRuns)
                allStdbyTime.append(stdbyTime)
                allStdbyRuns.append(stdbyRuns)
                allSembyTime.append(sembyRuns)
                allSembyRuns.append(sembyRuns)

            elif webdir == 'yes':
                meanWait = pd.read_hdf(
                    "%s\\%s_TimesMeanGroup.h5" % (directories[dir], os.path.basename(directories[dir])))
                semWait = pd.read_hdf("%s\\%s_TimesSemGroup.h5" % (directories[dir], os.path.basename(directories[dir])))

                allMeanWait.append(meanWait)
                allSemWait.append(semWait)


            # Check which day of APA Characterise experiments data is from
            ############################################################################################################
            ##################### THIS CAN NOW BE GOT FROM GET_EXP_DETAILS FUNCTION IN UTILS.PY ########################
            ############################################################################################################
            exp = Utils().get_exp_details(directories[dir])
            condition = exp['condition']
            speed = exp['Acspeed']
            perceivedspeed = exp['Pdspeed']
            VMTcon = exp['VMT condition']
            VMTtype = exp['VMT type']
            label = exp['plt label']


            allConditions.append(condition)
            allSpeeds.append(speed)
            allVMTcons.append(VMTcon)
            allVMTtypes.append(VMTtype)
            alllabels.append(label)


        if webdir == 'no':
            if error == 'std':
                allStdbyRuns = allStdbyRuns
            elif error == 'sem':
                allStdbyRuns = allSembyRuns

        ## plot 'byTime' data, ie plot line graphs of each measure and each skeleton for baseline, APA and washout
        #################################

        ## plot 'byRuns' data, ie on each plot there is a line graph with runs on x axis and all measures (seperately) on y axis. Do a seperate graph for each measure, skeleton and speed direction. Plot multiple speeds (of same direction) on a single plot.
        colors = Utils().get_cmap(n=4, name=colormap)
        if webdir == 'no':
            for m in measure:
                if m == 'length' or m == 'orientation':
                    for skelIdx, skel in enumerate(skeletonList):
                        # set parameters for fast to slow fig
                        plt.figure(num="%s, %s, Fast" % (m, skel), figsize=(11, 4))
                        axfast = plt.subplot(111)
                        axfast.set_xlim(0, max(runs))
                        if m == 'orientation':
                            axfast.set_ylim(0, 300)
                            patchwidth = 40
                        if m == 'length':
                            axfast.set_ylim(0, 650)
                            patchwidth = 70
                        plt.tick_params(
                            axis='x',  # changes apply to the x-axis
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            labelbottom=False)
                        axfast.add_patch(
                            plt.Rectangle((0, -patchwidth), 10, patchwidth, facecolor='lightblue', clip_on=False,
                                          linewidth=0))
                        axfast.add_patch(
                            plt.Rectangle((10, -patchwidth), 15, patchwidth, facecolor='lightskyblue', clip_on=False,
                                          linewidth=0))
                        axfast.add_patch(
                            plt.Rectangle((25, -patchwidth), 10, patchwidth, facecolor='dodgerblue', clip_on=False,
                                          linewidth=0))
                        axfast.text(0.1, -patchwidth + patchwidth * 0.1, 'Baseline = 16 cm/s\nTrials = 10', fontsize=15, zorder=5,
                                   color='k')
                        axfast.text(13, -patchwidth + patchwidth * 0.1, 'Visuomotor transformation\nTrials = 15', fontsize=15, zorder=5,
                                   color='k')
                        axfast.text(25.5, -patchwidth + patchwidth * 0.1, 'Washout 16 cm/s\nTrials = 10', fontsize=15, zorder=5,
                                   color='k')

                        # set params for slow to fast fig
                        plt.figure(num="%s, %s, Slow" % (m, skel), figsize=(11, 4))
                        axslow = plt.subplot(111)
                        axslow.set_xlim(0, max(runs))
                        if m == 'orientation':
                            axslow.set_ylim(0, 300)
                            patchwidth = 40
                        if m == 'length':
                            axslow.set_ylim(0, 600)
                            patchwidth = 70
                        plt.tick_params(
                            axis='x',  # changes apply to the x-axis
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            labelbottom=False)
                        axslow.add_patch(
                            plt.Rectangle((0, -patchwidth), 10, patchwidth, facecolor='lightblue', clip_on=False,
                                          linewidth=0, ))
                        axslow.add_patch(
                            plt.Rectangle((10, -patchwidth), 15, patchwidth, facecolor='lightskyblue', clip_on=False,
                                          linewidth=0))
                        axslow.add_patch(
                            plt.Rectangle((25, -patchwidth), 10, patchwidth, facecolor='dodgerblue', clip_on=False,
                                          linewidth=0))
                        axslow.text(0.1, -patchwidth + patchwidth * 0.1, 'Baseline = 4cm/s\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')
                        axslow.text(13, -patchwidth + patchwidth * 0.1, 'Visuomotor transformation\nTrials = 20', fontsize=15, zorder=5,
                                   color='k')
                        axslow.text(25.5, -patchwidth + patchwidth * 0.1, 'Washout = 4cm/s\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')


                        # ax.set_ylim(0, 250)
                        nfast = 0
                        nslow = 0
                        for VMTcIdx, VMTc in enumerate(allVMTcons):

                            # Fast to Slow
                            if VMTc == 'Fast':
                                plt.figure(num="%s, %s, Fast" % (m, skel))
                                plt.title('Fast - %s - %s' % (skel, m))
                                axfast.plot(allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m].index.values + 1,
                                           allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m], label=alllabels[VMTcIdx],
                                           color=colors(nfast))
                                axfast.fill_between(allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m].index.values + 1,
                                                   allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m] -
                                                   allStdbyRuns[VMTcIdx].loc(axis=1)[skel, m],
                                                   allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m] +
                                                   allStdbyRuns[VMTcIdx].loc(axis=1)[skel, m],
                                                   interpolate=False, alpha=0.1, color=colors(nfast))

                                nfast += 1  # MUST BE LAST

                            elif VMTc == 'Slow':
                                plt.figure(num="%s, %s, Slow" % (m, skel))
                                plt.title('Slow - %s - %s' % (skel, m))
                                plt.plot(allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m].index.values + 1,
                                         allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m], label=alllabels[VMTcIdx],
                                         color=colors(nslow))
                                plt.fill_between(allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m].index.values + 1,
                                                 allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m] -
                                                 allStdbyRuns[VMTcIdx].loc(axis=1)[skel, m],
                                                 allMeanbyRuns[VMTcIdx].loc(axis=1)[skel, m] +
                                                 allStdbyRuns[VMTcIdx].loc(axis=1)[skel, m],
                                                 interpolate=False, alpha=0.1, color=colors(nslow))

                                nslow += 1  # MUST BE LAST


                        if error == 'std':
                            plt.figure(num="%s, %s, Fast" % (m, skel))
                            plt.legend(title='Belt speed during VMT trials')
                            plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Fast\\fast_%s_%s.png" % (destfolder, skel, m),
                                        format='png')

                            plt.figure(num="%s, %s, Slow" % (m, skel))
                            plt.legend(title='Belt speed during VMT trials')
                            plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Slow\\slow_%s_%s.png" % (destfolder, skel, m),
                                        format='png')

                        elif error == 'sem':
                            plt.figure(num="%s, %s, Fast" % (m, skel))
                            plt.legend(title='Belt speed during VMT trials')
                            plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Fast\\fast_%s_%s_SEM.png" % (destfolder, skel, m),
                                        format='png')

                            plt.figure(num="%s, %s, Slow" % (m, skel))
                            plt.legend(title='Belt speed during VMT trials')
                            plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Slow\\slow_%s_%s_SEM.png" % (destfolder, skel, m),
                                        format='png')


                if m == 'triangle height':
                    patchwidth = 30
                    plt.figure(num="%s, Fast" % m, figsize=(11, 4))
                    axfast = plt.subplot(111)
                    axfast.set_xlim(0, max(runs))
                    axfast.set_ylim(0, 200)
                    plt.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)
                    axfast.add_patch(plt.Rectangle((0, -patchwidth), 10, patchwidth, facecolor='lightblue', clip_on=False,
                                                  linewidth=0))
                    axfast.add_patch(
                        plt.Rectangle((10, -patchwidth), 15, patchwidth, facecolor='lightskyblue', clip_on=False,
                                      linewidth=0))
                    axfast.add_patch(
                        plt.Rectangle((25, -patchwidth), 10, patchwidth, facecolor='dodgerblue', clip_on=False,
                                      linewidth=0))
                    axfast.text(0.1, -patchwidth + patchwidth * 0.2, 'Baseline = 16 cm/s\nTrials = 5', fontsize=15, zorder=5,
                               color='k')
                    axfast.text(13, -patchwidth + patchwidth * 0.2, 'Visuomotor Transformation\nTrials = 20', fontsize=15, zorder=5,
                               color='k')
                    axfast.text(25.1, -patchwidth + patchwidth * 0.2, 'Washout = 16 cm/s\nTrials = 5', fontsize=15, zorder=5,
                               color='k')

                    plt.figure(num="%s, Slow" % m, figsize=(11, 4))
                    axslow = plt.subplot(111)
                    axslow.set_xlim(0, max(runs))
                    axslow.set_ylim(0, 200)
                    plt.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)
                    axslow.add_patch(plt.Rectangle((0, -patchwidth), 10, patchwidth, facecolor='lightblue', clip_on=False,
                                                  linewidth=0, ))
                    axslow.add_patch(
                        plt.Rectangle((10, -patchwidth), 15, patchwidth, facecolor='lightskyblue', clip_on=False,
                                      linewidth=0))
                    axslow.add_patch(
                        plt.Rectangle((25, -patchwidth), 10, patchwidth, facecolor='dodgerblue', clip_on=False,
                                      linewidth=0))
                    axslow.text(0.1, -patchwidth + patchwidth * 0.2, 'Baseline = 4 cm/s\nTrials = 5', fontsize=15, zorder=5,
                               color='k')
                    axslow.text(13, -patchwidth + patchwidth * 0.2, 'Visuomotor Transformation\nTrials = 20', fontsize=15, zorder=5,
                               color='k')
                    axslow.text(25.1, -patchwidth + patchwidth * 0.2, 'Washout = 4 cm/s\nTrials = 5', fontsize=15, zorder=5,
                               color='k')


                    # ax.set_ylim(0, 250)
                    nfast = 0
                    nslow = 0

                    for VMTcIdx, VMTc in enumerate(allVMTcons):

                        # Fast to Slow
                        if VMTc == 'Fast':
                            plt.figure(num="%s, Fast" % m)
                            plt.title('Fast - %s' % m)
                            axfast.plot(allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m].index.values + 1,
                                       allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m], label=alllabels[VMTcIdx],
                                       color=colors(nfast))
                            axfast.fill_between(allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m].index.values + 1,
                                               allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m] -
                                               allStdbyRuns[VMTcIdx].loc(axis=1)['combined', m],
                                               allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m] +
                                               allStdbyRuns[VMTcIdx].loc(axis=1)['combined', m],
                                               interpolate=False, alpha=0.1, color=colors(nfast))

                            nfast += 1  # MUST BE LAST

                        elif VMTc == 'Slow':
                            plt.figure(num="%s, Slow" % m)
                            plt.title('Slow - %s' % m)
                            axslow.plot(allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m].index.values + 1,
                                     allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m], label=alllabels[VMTcIdx],
                                     color=colors(nslow))
                            axslow.fill_between(allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m].index.values + 1,
                                             allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m] -
                                             allStdbyRuns[VMTcIdx].loc(axis=1)['combined', m],
                                             allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', m] +
                                             allStdbyRuns[VMTcIdx].loc(axis=1)['combined', m],
                                             interpolate=False, alpha=0.1, color=colors(nslow))
                            nslow += 1  # MUST BE LAST



                    if error == 'std':
                        plt.figure(num="%s, Fast" % m)
                        plt.legend(title='Belt speed during VMT trials')
                        plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Fast\\fast_%s.png" % (destfolder, m),
                                    format='png')

                        plt.figure(num="%s, Slow" % m)
                        plt.legend(title='Belt speed during VMT trials')
                        plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Slow\\slow_%s.png" % (destfolder, m),
                                    format='png')

                    elif error == 'sem':
                        plt.figure(num="%s, Fast" % m)
                        plt.legend(title='Belt speed during VMT trials')
                        plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Fast\\fast_%s_SEM.png" % (destfolder, m),
                                    format='png')

                        plt.figure(num="%s, Slow" % m)
                        plt.legend(title='Belt speed during VMT trials')
                        plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Slow\\slow_%s_SEM.png" % (destfolder, m),
                                    format='png')

                if m == 'triangle angles':
                    angles = ['nose angle', 'hump angle', 'tail1 angle']
                    for aIdx, a in enumerate(angles):
                        plt.figure(num="%s, Fast" % a, figsize=(11, 4))
                        axfast = plt.subplot(111)
                        axfast.set_xlim(0, max(runs))
                        if a == 'hump angle':
                            axfast.set_ylim(0, 175)
                            patchwidth = 20
                        else:
                            axfast.set_ylim(0, 80)
                            patchwidth = 10
                        plt.tick_params(
                            axis='x',  # changes apply to the x-axis
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            labelbottom=False)
                        axfast.add_patch(
                            plt.Rectangle((0, -patchwidth), 10, patchwidth, facecolor='lightblue', clip_on=False,
                                          linewidth=0))
                        axfast.add_patch(
                            plt.Rectangle((10, -patchwidth), 15, patchwidth, facecolor='lightskyblue', clip_on=False,
                                          linewidth=0))
                        axfast.add_patch(
                            plt.Rectangle((25, -patchwidth), 10, patchwidth, facecolor='dodgerblue', clip_on=False,
                                          linewidth=0))
                        axfast.text(0.1, -patchwidth + patchwidth * 0.1, 'Baseline = 16 cm/s\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')
                        axfast.text(13, -patchwidth + patchwidth * 0.1, 'Visuomotor transformation\nTrials = 20', fontsize=15, zorder=5,
                                   color='k')
                        axfast.text(25.5, -patchwidth + patchwidth * 0.1, 'Washout = 16cm/s\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')

                        plt.figure(num="%s, Slow" % a, figsize=(11, 4))
                        axslow = plt.subplot(111)
                        axslow.set_xlim(0, max(runs))
                        if a == 'hump angle':
                            axslow.set_ylim(0, 175)
                            patchwidth = 20
                        else:
                            axslow.set_ylim(0, 80)
                            patchwidth = 10
                        plt.tick_params(
                            axis='x',  # changes apply to the x-axis
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            labelbottom=False)
                        axslow.add_patch(
                            plt.Rectangle((0, -patchwidth), 10, patchwidth, facecolor='lightblue', clip_on=False,
                                          linewidth=0, ))
                        axslow.add_patch(
                            plt.Rectangle((10, -patchwidth), 15, patchwidth, facecolor='lightskyblue', clip_on=False,
                                          linewidth=0))
                        axslow.add_patch(
                            plt.Rectangle((25, -patchwidth), 10, patchwidth, facecolor='dodgerblue', clip_on=False,
                                          linewidth=0))
                        axslow.text(0.1, -patchwidth + patchwidth * 0.1, ' Baseline = 4cm/s\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')
                        axslow.text(13, -patchwidth + patchwidth * 0.1, 'Visuomotor transformation\nTrials = 20', fontsize=15, zorder=5,
                                   color='k')
                        axslow.text(25.5, -patchwidth + patchwidth * 0.1, 'Washout = 4cm/s\nTrials = 5', fontsize=15, zorder=5,
                                   color='k')


                        # ax.set_ylim(0, 250)
                        nfast = 0
                        nslow = 0

                        for VMTcIdx, VMTc in enumerate(allVMTcons):

                            # Fast to Slow
                            if VMTc == 'Fast':
                                plt.figure(num="%s, Fast" % a)
                                plt.title('Fast - %s' % a)
                                axfast.plot(allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a].index.values + 1,
                                           allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a], label=alllabels[VMTcIdx],
                                           color=colors(nfast))
                                axfast.fill_between(allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a].index.values + 1,
                                                   allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a] -
                                                   allStdbyRuns[VMTcIdx].loc(axis=1)['combined', a],
                                                   allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a] +
                                                   allStdbyRuns[VMTcIdx].loc(axis=1)['combined', a],
                                                   interpolate=False, alpha=0.1, color=colors(nfast))

                                nfast += 1  # MUST BE LAST

                            elif VMTc == 'Slow':
                                plt.figure(num="%s, Slow" % a)
                                plt.title('Slow - %s' % a)
                                plt.plot(allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a].index.values + 1,
                                         allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a], label=alllabels[VMTcIdx],
                                         color=colors(nslow))
                                plt.fill_between(allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a].index.values + 1,
                                                 allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a] -
                                                 allStdbyRuns[VMTcIdx].loc(axis=1)['combined', a],
                                                 allMeanbyRuns[VMTcIdx].loc(axis=1)['combined', a] +
                                                 allStdbyRuns[VMTcIdx].loc(axis=1)['combined', a],
                                                 interpolate=False, alpha=0.1, color=colors(nslow))

                                nslow += 1  # MUST BE LAST


                        if error == 'std':
                            plt.figure(num="%s, Fast" % a)
                            plt.legend(title='Belt speed during VMT trials')
                            plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Fast\\fast_%s.png" % (destfolder, a),
                                        format='png')

                            plt.figure(num="%s, Slow" % a)
                            plt.legend(title='Belt speed during VMT trials')
                            plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Slow\\slow_%s.png" % (destfolder, a),
                                        format='png')


                        elif error == 'sem':
                            plt.figure(num="%s, Fast" % a)
                            plt.legend(title='Belt speed during VMT trials')
                            plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Fast\\fast_%s_SEM.png" % (destfolder, a),
                                        format='png')

                            plt.figure(num="%s, Slow" % a)
                            plt.legend(title='Belt speed during VMT trials')
                            plt.savefig("%s\\plots\\APA_VMT\\byRuns\\Slow\\slow_%s_SEM.png" % (destfolder, a),
                                        format='png')

        elif webdir == 'yes':
            plt.figure(num="Wait time (s), Fast", figsize=(11, 4))
            axfast = plt.subplot(111)
            axfast.set_xlim(0, max(runs))
            axfast.set_ylim(0, 80)
            axfast.set_ylabel('Wait time (s)')
            patchwidth = 10
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
            axfast.add_patch(
                plt.Rectangle((0, -patchwidth), 10, patchwidth, facecolor='lightblue', clip_on=False,
                              linewidth=0))
            axfast.add_patch(
                plt.Rectangle((10, -patchwidth), 15, patchwidth, facecolor='lightskyblue', clip_on=False,
                              linewidth=0))
            axfast.add_patch(
                plt.Rectangle((25, -patchwidth), 10, patchwidth, facecolor='dodgerblue', clip_on=False,
                              linewidth=0))
            axfast.text(0.1, -patchwidth + patchwidth * 0.1, 'Baseline = 16 cm/s\nTrials = 5', fontsize=15, zorder=5,
                        color='k')
            axfast.text(13, -patchwidth + patchwidth * 0.1, 'Visuomotor transformation\nTrials = 20', fontsize=15,
                        zorder=5,
                        color='k')
            axfast.text(25.5, -patchwidth + patchwidth * 0.1, 'Washout = 16cm/s\nTrials = 5', fontsize=15, zorder=5,
                        color='k')

            plt.figure(num="Wait time (s), Slow", figsize=(11, 4))
            axslow = plt.subplot(111)
            axslow.set_xlim(0, max(runs))
            axslow.set_ylim(0, 80)
            axslow.set_ylabel('Wait time (s)')
            patchwidth = 10
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
            axslow.add_patch(
                plt.Rectangle((0, -patchwidth), 10, patchwidth, facecolor='lightblue', clip_on=False,
                              linewidth=0, ))
            axslow.add_patch(
                plt.Rectangle((10, -patchwidth), 15, patchwidth, facecolor='lightskyblue', clip_on=False,
                              linewidth=0))
            axslow.add_patch(
                plt.Rectangle((25, -patchwidth), 10, patchwidth, facecolor='dodgerblue', clip_on=False,
                              linewidth=0))
            axslow.text(0.1, -patchwidth + patchwidth * 0.1, ' Baseline = 4cm/s\nTrials = 5', fontsize=10, zorder=5,
                        color='k')
            axslow.text(13, -patchwidth + patchwidth * 0.1, 'Visuomotor transformation\nTrials = 20', fontsize=10,
                        zorder=5,
                        color='k')
            axslow.text(25.5, -patchwidth + patchwidth * 0.1, 'Washout = 4cm/s\nTrials = 5', fontsize=10, zorder=5,
                        color='k')

            # ax.set_ylim(0, 250)
            nfast = 0
            nslow = 0

            for VMTcIdx, VMTc in enumerate(allVMTcons):

                # Fast to Slow
                if VMTc == 'Fast':
                    plt.figure(num="Wait time (s), Fast")
                    plt.title('Fast - Wait time')
                    axfast.plot(allMeanWait[VMTcIdx].index.values + 1,
                                allMeanWait[VMTcIdx].loc(axis=1)['Wait Time (s)'], label=alllabels[VMTcIdx],
                                color=colors(nfast))
                    axfast.fill_between(allMeanWait[VMTcIdx].index.values + 1,
                                        allMeanWait[VMTcIdx].loc(axis=1)['Wait Time (s)'] -
                                        allSemWait[VMTcIdx].loc(axis=1)['Wait Time (s)'],
                                        allMeanWait[VMTcIdx].loc(axis=1)['Wait Time (s)'] +
                                        allSemWait[VMTcIdx].loc(axis=1)['Wait Time (s)'],
                                        interpolate=False, alpha=0.1, color=colors(nfast))

                    nfast += 1  # MUST BE LAST

                elif VMTc == 'Slow':
                    plt.figure(num="Wait time (s), Slow")
                    plt.title('Fast - Wait time')
                    axslow.plot(allMeanWait[VMTcIdx].index.values + 1,
                                allMeanWait[VMTcIdx].loc(axis=1)['Wait Time (s)'], label=alllabels[VMTcIdx],
                                color=colors(nfast))
                    axslow.fill_between(allMeanWait[VMTcIdx].index.values + 1,
                                        allMeanWait[VMTcIdx].loc(axis=1)['Wait Time (s)'] -
                                        allSemWait[VMTcIdx].loc(axis=1)['Wait Time (s)'],
                                        allMeanWait[VMTcIdx].loc(axis=1)['Wait Time (s)'] +
                                        allSemWait[VMTcIdx].loc(axis=1)['Wait Time (s)'],
                                        interpolate=False, alpha=0.1, color=colors(nfast))

                    nslow += 1  # MUST BE LAST


            plt.figure(num="Wait time (s), Fast")
            plt.legend(title='Belt speed during VMT trials')
            plt.savefig("%s\\plots\\APA_VMT\\WaitTime(webcam)\\Fast\\fast_WaitTime_SEM.png" % destfolder,
                        format='png')

            plt.figure(num="Wait time (s), Slow")
            plt.legend(title='Belt speed during VMT trials')
            plt.savefig("%s\\plots\\APA_VMT\\WaitTime(webcam)\\Slow\\slow_WaitTime_SEM.png" % destfolder,
                        format='png')





#
#
#
#
# # for APA Char

APACharDirectories = [r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201204",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201201",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201208",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201209",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201202",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201207",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201203",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201210",
               #r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201130",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201211"]
VMTDirectories = [r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201214",
               #r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201215",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201216",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201217",
               r"M:\DLC_DualBelt-Holly-2021-02-11\analysed_data\iteration-2\20201218"]

# exp = []
            # runPhases = []
            # indexList = []
            # if '20201130' in files[df][0]:
            #     exp = 'APACharBaseline'
            #     runPhases = [list(range(0, 20))]
            #     indexList = ['BaselineRuns']
            # elif '20201201' in files[df][0]:
            #     exp = 'APACharNoWash'
            #     runPhases = [list(range(0, 5)), list(range(5, 20))]
            #     indexList = ['BaselineRuns', 'APARuns']
            # elif '20201202' in files[df][0]:
            #     exp = 'APACharNoWash'
            #     runPhases = [list(range(0, 5)), list(range(5, 20))]
            #     indexList = ['BaselineRuns', 'APARuns']
            # elif '20201203' in files[df][0]:
            #     exp = 'APACharNoWash'
            #     runPhases = [list(range(0, 5)), list(range(5, 20))]
            #     indexList = ['BaselineRuns', 'APARuns']
            # elif '20201204' in files[df][0]:
            #     exp = 'APACharNoWash'
            #     runPhases = [list(range(0, 5)), list(range(5, 20))]
            #     indexList = ['BaselineRuns', 'APARuns']
            # elif '20201207' in files[df][0]:
            #     exp = 'APAChar'
            #     runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            #     indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            # elif '20201208' in files[df][0]:
            #     exp = 'APAChar'
            #     runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            #     indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            # elif '20201209' in files[df][0]:
            #     exp = 'APAChar'
            #     runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            #     indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            # elif '20201210' in files[df][0]:
            #     exp = 'APAChar'
            #     runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            #     indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            # elif '20201211' in files[df][0]:
            #     exp = 'APAChar'
            #     runPhases = [list(range(0, 5)), list(range(5, 20)), list(range(20, 25))]
            #     indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            # elif '20201214' in files[df][0]:
            #     exp = 'VisuoMotTransf'
            #     runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            #     indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            # elif '20201215' in files[df][0]:
            #     exp = 'VisuoMotTransf'
            #     runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            #     indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            # elif '20201216' in files[df][0]:
            #     exp = 'VisuoMotTransf'
            #     runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            #     indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            # elif '20201217' in files[df][0]:
            #     exp = 'VisuoMotTransf'
            #     runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            #     indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            # else:
            #     print('Somethings gone wrong, cannot find this file')


# if os.path.basename(directories[dir]) == '20201130':
#     condition = 'Control'
#     speed = 0
# elif os.path.basename(directories[dir]) == '20201201':
#     condition = 'FastToSlow'
#     speed = 0.08
# elif os.path.basename(directories[dir]) == '20201204':
#     condition = 'FastToSlow'
#     speed = 0.04
# elif os.path.basename(directories[dir]) == '20201208':
#     condition = 'FastToSlow'
#     speed = 0.16
# elif os.path.basename(directories[dir]) == '20201209':
#     condition = 'FastToSlow'
#     speed = 0.32
# elif os.path.basename(directories[dir]) == '20201202':
#     condition = 'SlowToFast'
#     speed = 0.04
# elif os.path.basename(directories[dir]) == '20201203':
#     condition = 'SlowToFast'
#     speed = 0.16
# elif os.path.basename(directories[dir]) == '20201207':
#     condition = 'SlowToFast'
#     speed = 0.08
# elif os.path.basename(directories[dir]) == '20201210':
#     condition = 'SlowToFast'
#     speed = 0.32
# elif os.path.basename(directories[dir]) == '20201211':
#     condition = 'PerceptionTest'
#     speed = 'actual = 0.16, perceived = 1.00'
# else:
#     print('something wrong')