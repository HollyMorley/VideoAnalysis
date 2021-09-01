## GetTimes.py
## gets time-related information from webcam data

from GetRuns import GetRuns
from GetWebcamRuns import GetWebcamRuns
from Config import *
from utils import Utils
from scipy import signal
import pandas as pd
import numpy as np
import os
from glob import glob

class GetTimes:
    def __init__(self):
        super().__init__()

    def frameToTimeWeb(self, files=None, directory=None):  # written for webcam stuff
        print('start frameToTimeWeb')
        data = Utils().Getlistoffiles(files=files,directory=directory,scorer=scorerWeb)
        runsALL = list()
        for d in range(0,len(data)):
            df = pd.read_hdf(data[d])
            runs = GetWebcamRuns().findRuns(Dataframe=df)
            frameIdx = runs.index.get_level_values(level=0)
            runIdx = runs.index.get_level_values(level=1)
            timeIdx = (runs.index.get_level_values(level=0)/webcamFPS)*webcamWrongMultiple  # corresponding seconds from beginning of video for each frame
            #runs.index = runs.index.set_levels(timeIdx, level=0).set_names('Time(s)', level=0)
            runs = runs.set_index([frameIdx, timeIdx, runIdx])
            runs.index = runs.index.set_names(['Frame', 'Time(s)'], level=[0, 1])
            runsALL.append(runs)
        return runsALL

    def getTimesWeb(self, files=None, directory=None):
        print('start getTimesWeb')
        filenames = Utils().Getlistoffiles(files=files, directory=directory,scorer=scorerWeb)
        data = self.frameToTimeWeb(files=files, directory=directory)

        timesALL = list()
        for i in range(0, len(data)):
            print(i)

            # get end of trial times
            end = np.array(data[i].loc[data[i]['Quadrants'] == 4].index.get_level_values(level='Time(s)'))
            mask = abs(end - np.roll(end,1)) > 15

            # get trial time, run time, wait time, run speed
            #trialTime = np.array(data[i].loc[data[i]['RunEnd'] == 1].index.get_level_values(level='Time(s)') - data[i].loc[data[i]['TrialStart'] == 1].index.get_level_values(level='Time(s)'))
            #runTime = np.array(data[i].loc[data[i]['RunEnd'] == 1].index.get_level_values(level='Time(s)') - data[i].loc[data[i]['RunStart'] == 1].index.get_level_values(level='Time(s)'))
            trialTime = np.array(end[mask] - data[i].loc[data[i]['TrialStart'] == 1].index.get_level_values(level='Time(s)'))
            runTime = np.array(end[mask] - data[i].loc[data[i]['RunStart'] == 1].index.get_level_values(level='Time(s)'))
            waitTime = np.array(end[mask] - data[i].loc[data[i]['TrialStart'] == 1].index.get_level_values(level='Time(s)'))

            # get time spent in each quadrant (s)
            #q1Time = list(data[i].loc[data[i]['Quadrants'] == 1].index.get_level_values(level='Time(s)') - data[i].loc[data[i]['TrialStart'] == 1].index.get_level_values(level='Time(s)'))
            q1TimeALL = list()
            q2TimeALL = list()
            q3TimeALL = list()
            q4TimeALL = list()
            for r in range(0, len(data[i].index.unique(level='Run'))):
                #if np.sum(data[i].loc(axis=1)['Quadrants'].xs(r, axis=0, level='Run') == 1) > 0:
                try:
                    if 2 in data[i].loc(axis=1)['Quadrants'].xs(r, axis=0, level='Run').values:
                        q1Time = np.array(data[i].loc[data[i]['Quadrants'] == 2].xs(r, axis=0, level='Run').head(1).index.get_level_values(level='Time(s)'))[0] - np.array(data[i].loc[data[i]['RunStart'] == 1].xs(r, axis=0, level='Run').index.get_level_values(level='Time(s)'))[0]
                    if all(x in data[i].loc(axis=1)['Quadrants'].xs(r, axis=0, level='Run').values for x in [2,3]):
                        q2Time = np.array(data[i].loc[data[i]['Quadrants'] == 3].xs(r, axis=0, level='Run').head(1).index.get_level_values(level='Time(s)'))[0] - np.array(data[i].loc[data[i]['Quadrants'] == 2].xs(r, axis=0, level='Run').head(1).index.get_level_values(level='Time(s)'))[0]
                    if all(x in data[i].loc(axis=1)['Quadrants'].xs(r, axis=0, level='Run').values for x in [3,4]):
                        q3Time = np.array(data[i].loc[data[i]['Quadrants'] == 4].xs(r, axis=0, level='Run').head(1).index.get_level_values(level='Time(s)'))[0] - np.array(data[i].loc[data[i]['Quadrants'] == 3].xs(r, axis=0, level='Run').head(1).index.get_level_values(level='Time(s)'))[0]
                    if 4 in data[i].loc(axis=1)['Quadrants'].xs(r, axis=0, level='Run').values:
                        q4endIdx = np.array(data[i].loc[data[i]['Quadrants'] == 4].xs(r, axis=0, level='Run').tail(1).index.get_level_values(level='Frame'))[0]
                        try:
                            q4Time = np.array(data[i].xs(r, axis=0, level='Run').loc(axis=0)[q4endIdx+1].index)[0] - np.array(data[i].loc[data[i]['Quadrants'] == 4].xs(r, axis=0, level='Run').head(1).index.get_level_values(level='Time(s)'))[0]
                        except:
                            q4Time = np.array(data[i].xs(r, axis=0, level='Run').loc(axis=0)[q4endIdx].index)[0] - np.array(data[i].loc[data[i]['Quadrants'] == 4].xs(r, axis=0, level='Run').head(1).index.get_level_values(level='Time(s)'))[0] # for when the last value of q4 is the same as end of run
                except:
                    print('There is a problem finding times for the following runs:')
                    print(r)

                q1TimeALL.append(q1Time)
                q2TimeALL.append(q2Time)
                q3TimeALL.append(q3Time)
                q4TimeALL.append(q4Time)

            q1TimeALL = np.array(q1TimeALL)
            q2TimeALL = np.array(q2TimeALL)
            q3TimeALL = np.array(q3TimeALL)
            q4TimeALL = np.array(q4TimeALL)

            # Run speed (both broad speed and try to get actual mouse speed based on belt speed) (cm/s)
            TotalRunSpeed = midBeltLength/runTime
            TotalQ1Speed = midBeltLengthQuad/q1TimeALL
            TotalQ2Speed = midBeltLengthQuad/q2TimeALL
            TotalQ3Speed = midBeltLengthQuad/q3TimeALL
            TotalQ4Speed = midBeltLengthQuad/q4TimeALL ##### TAKE THIS WITH A PINCH OF SALT: this is slow partly because the end of this quadrant is when Tail1 has gone off belt. Therefore, this includes the time where the mouse stabilises on the stationary platform with front paws.


            # find actual mouse run speed by taking away belt speed (in trials where belt moves)
            exp = Utils().get_exp_details(filenames[i])

            MouseRunSpeed = TotalRunSpeed
            MouseQ1Speed = TotalQ1Speed
            MouseQ2Speed = TotalQ2Speed
            MouseQ3Speed = TotalQ3Speed
            MouseQ4Speed = TotalQ4Speed

            # overwrite speeds in trials where belt is moving with mouse speeds where belt speed is subtracted
            if exp['exp'] == 'APACharNoWash':
                if data[i].index.get_level_values(level='Run')[-1] == exp['runPhases'][1][-1]:
                    MouseRunSpeed[exp['runPhases'][1]] = TotalRunSpeed[exp['runPhases'][1]] - exp['Acspeed']
                    MouseQ1Speed[exp['runPhases'][1]] = TotalQ1Speed[exp['runPhases'][1]] - exp['Acspeed']
                    MouseQ2Speed[exp['runPhases'][1]] = TotalQ2Speed[exp['runPhases'][1]] - exp['Acspeed']
                    MouseQ3Speed[exp['runPhases'][1]] = TotalQ3Speed[exp['runPhases'][1]] - exp['Acspeed']
                    MouseQ4Speed[exp['runPhases'][1]] = TotalQ4Speed[exp['runPhases'][1]] - exp['Acspeed']
                else:
                    index = np.linspace(exp['runPhases'][1][0], data[i].index.get_level_values(level='Run')[-1], num=int(data[i].index.get_level_values(level='Run')[-1] - exp['runPhases'][1][0] +1), dtype=int)
                    MouseRunSpeed[index] = TotalRunSpeed[index] - exp['Acspeed']
                    MouseQ1Speed[index] = TotalQ1Speed[index] - exp['Acspeed']
                    MouseQ2Speed[index] = TotalQ2Speed[index] - exp['Acspeed']
                    MouseQ3Speed[index] = TotalQ3Speed[index] - exp['Acspeed']
                    MouseQ4Speed[index] = TotalQ4Speed[index] - exp['Acspeed']
            elif exp['exp'] == 'APAChar':
                if data[i].index.get_level_values(level='Run')[-1] == exp['runPhases'][2][-1]:
                    print("have the correct amount of runs!")
                    MouseRunSpeed[exp['runPhases'][1]] = TotalRunSpeed[exp['runPhases'][1]] - exp['Acspeed']
                    MouseQ1Speed[exp['runPhases'][1]] = TotalQ1Speed[exp['runPhases'][1]] - exp['Acspeed']
                    MouseQ2Speed[exp['runPhases'][1]] = TotalQ2Speed[exp['runPhases'][1]] - exp['Acspeed']
                    MouseQ3Speed[exp['runPhases'][1]] = TotalQ3Speed[exp['runPhases'][1]] - exp['Acspeed']
                    MouseQ4Speed[exp['runPhases'][1]] = TotalQ4Speed[exp['runPhases'][1]] - exp['Acspeed']
                else:
                    #if filenames[i] == 'M:\\DLC_DualBelt_webcam-Holly-2021-04-22\\analysed_data\\iteration-2\\20201211\\20201211MR_webcamDLC_resnet50_DLC_DualBelt_webcamApr22shuffle1_800000.h5':
                     #   index = np.linspace(exp['runPhases'][1][0], data[i].index.get_level_values(level='Run')[-1] - 5, num=int((data[i].index.get_level_values(level='Run')[-1] - 5) - exp['runPhases'][1][0] + 1), dtype=int)
                    if filenames[i] == 'M:\\DLC_DualBelt_webcam-Holly-2021-04-22\\analysed_data\\iteration-2\\20201211\\20201211FR_webcamDLC_resnet50_DLC_DualBelt_webcamApr22shuffle1_800000.h5':
                        index = np.linspace(exp['runPhases'][1][0], data[i].index.get_level_values(level='Run')[-1], num=int((data[i].index.get_level_values(level='Run')[-1]) - exp['runPhases'][1][0] + 1), dtype=int)
                    if filenames[i] == 'M:\\DLC_DualBelt_webcam-Holly-2021-04-22\\analysed_data\\iteration-2\\20201207\\20201207FLDLC_resnet50_DLC_DualBelt_webcamApr22shuffle1_800000.h5':
                        index = np.linspace(exp['runPhases'][1][0], data[i].index.get_level_values(level='Run')[-1], num=int((data[i].index.get_level_values(level='Run')[-1]) - exp['runPhases'][1][0] + 1), dtype=int)
                    try:
                        MouseRunSpeed[index] = TotalRunSpeed[index] - exp['Acspeed']
                        MouseQ1Speed[index] = TotalQ1Speed[index] - exp['Acspeed']
                        MouseQ2Speed[index] = TotalQ2Speed[index] - exp['Acspeed']
                        MouseQ3Speed[index] = TotalQ3Speed[index] - exp['Acspeed']
                        MouseQ4Speed[index] = TotalQ4Speed[index] - exp['Acspeed']
                    except:
                        print("indexing for APAChar incorrect for %s" % filenames[i])


            elif exp['exp'] == 'VisuoMotTransf':
                VMTcons = ['AcBaselineSpeed', 'AcVMTSpeed', 'AcWashoutSpeed']
                if data[i].index.get_level_values(level='Run')[-1] == exp['runPhases'][2][-1]:
                    for num, con in enumerate(VMTcons):
                        MouseRunSpeed[exp['runPhases'][num]] = TotalRunSpeed[exp['runPhases'][num]] - exp[con]
                        MouseQ1Speed[exp['runPhases'][num]] = TotalQ1Speed[exp['runPhases'][num]] - exp[con]
                        MouseQ2Speed[exp['runPhases'][num]] = TotalQ2Speed[exp['runPhases'][num]] - exp[con]
                        MouseQ3Speed[exp['runPhases'][num]] = TotalQ3Speed[exp['runPhases'][num]] - exp[con]
                        MouseQ4Speed[exp['runPhases'][num]] = TotalQ4Speed[exp['runPhases'][num]] - exp[con]
                else:
                    if filenames[i] == 'M:\\DLC_DualBelt_webcam-Holly-2021-04-22\\analysed_data\\iteration-2\\20201215\\20201215FLR_webcamDLC_resnet50_DLC_DualBelt_webcamApr22shuffle1_800000.h5':
                        exp['runPhases'][1] = exp['runPhases'][1][0:-2]
                        exp['runPhases'][2] = []
                    if filenames[i] == 'M:\\DLC_DualBelt_webcam-Holly-2021-04-22\\analysed_data\\iteration-2\\20201216\\20201216FR_webcamDLC_resnet50_DLC_DualBelt_webcamApr22shuffle1_800000.h5':
                        exp['runPhases'][1] = exp['runPhases'][1][0:-4]
                        exp['runPhases'][2] = []
                    if filenames[i] == 'M:\\DLC_DualBelt_webcam-Holly-2021-04-22\\analysed_data\\iteration-2\\20201216\\20201218FR_webcamDLC_resnet50_DLC_DualBelt_webcamApr22shuffle1_800000.h5':
                        exp['runPhases'][2] = exp['runPhases'][2][0:-5]
                    try:
                        for num, con in enumerate(VMTcons):
                            MouseRunSpeed[exp['runPhases'][num]] = TotalRunSpeed[exp['runPhases'][num]] - exp[con]
                            MouseQ1Speed[exp['runPhases'][num]] = TotalQ1Speed[exp['runPhases'][num]] - exp[con]
                            MouseQ2Speed[exp['runPhases'][num]] = TotalQ2Speed[exp['runPhases'][num]] - exp[con]
                            MouseQ3Speed[exp['runPhases'][num]] = TotalQ3Speed[exp['runPhases'][num]] - exp[con]
                            MouseQ4Speed[exp['runPhases'][num]] = TotalQ4Speed[exp['runPhases'][num]] - exp[con]
                    except:
                        print("No runs for %s in file: %s" % (exp['runPhases'][num], filenames[i]))

            # make dictionary with all above and return
            times = {
                "Trial Time (s)": trialTime,
                "Run Time (s)": runTime,
                "Wait Time (s)": waitTime,
                "Q1 Time (s)": q1TimeALL,
                "Q2 Time (s)": q2TimeALL,
                "Q3 Time (s)": q3TimeALL,
                "Q4 Time (s)": q4TimeALL,
                "Run Speed (cm/s)": MouseRunSpeed,
                "Q1 Speed (cm/s)": MouseQ1Speed,
                "Q2 Speed (cm/s)": MouseQ2Speed,
                "Q3 Speed (cm/s)": MouseQ3Speed,
                "Q4 Speed (cm/s)": MouseQ4Speed
            }
            timesALL.append(times)

        #return list of dictionaries for all files passed
        return timesALL

    def getNudgesWeb(self, files=None, directory=None):
        print('start getNudgesWeb')
        runs = self.frameToTimeWeb(files=files, directory=directory)
        #nudgesALL = list()
        #nudges_countALL = list()
        nudgeDataALL = list()

        for r in range(0, len(runs)):
            HEIGHTS_fixed_run = []
            frame = []
            time = []
            Run = []
            for run in range(0, len(runs[r].index.unique(level='Run'))):
                x = runs[r].xs(run, axis=0, level='Run').loc(axis=1)['Pen', 'likelihood']
                peaks, _ = signal.find_peaks(x, height=(0.9999, 1), distance=50)
                peaksidx = x.iloc[peaks].index
                if len(peaksidx) != 0:
                    for i in range(len(peaksidx)):
                        HEIGHTS_fixed_run.append(x[peaksidx].tolist()[i])
                        frame.append(runs[r].xs(run, axis=0, level='Run').loc(axis=1)['Pen', 'likelihood'].index[i][0])
                        time.append(runs[r].xs(run, axis=0, level='Run').loc(axis=1)['Pen', 'likelihood'].index[i][1])
                        Run.append(run)
            nudges = pd.DataFrame()
            nudges['Heights'] = HEIGHTS_fixed_run
            nudges['frame'] = frame
            nudges['time'] = time
            nudges['Run'] = Run
            #nudgesALL.append(nudges)

            nudges_count = nudges.pivot_table(index=['Run'], aggfunc='size')
            #nudges_countALL.append(nudges_count)

            nudgeData = {
                "Nudges list": nudges,
                "Nudges count per run": nudges_count
            }
            nudgeDataALL.append(nudgeData)

        return nudgeDataALL


    def collateGroupWebData(self, files=None, directory=None):
        print('start collateGroupWebData')
        fnames = Utils().Getlistoffiles(files=files,directory=directory,scorer=scorerWeb)
        times = self.getTimesWeb(files=fnames,directory=None)
        nudges = self.getNudgesWeb(files=fnames,directory=None)
        exp = Utils().get_exp_details(file=fnames[0])  # all files will be from the same exp

        # reformat dictionaries into stack of dfs which i can then groupby
        timesGroup = pd.DataFrame()
        nudgesGroup = pd.DataFrame(columns=np.linspace(0,sum([len(r) for r in exp['runPhases']]),sum([len(r) for r in exp['runPhases']]),dtype=int))
        for i in range(0,len(fnames)):
            timesdf = pd.DataFrame.from_dict(times[i])
            nudgesdf = nudges[i]['Nudges count per run']
            nudgesdf.name = 'count per run'
            #timesGroup.append(timesdf)
            #nudgesGroup.append(nudgesdf)
            timesGroup = timesGroup.append(timesdf)
            nudgesGroup = nudgesGroup.append(nudgesdf)[np.linspace(0,sum([len(r) for r in exp['runPhases']]),sum([len(r) for r in exp['runPhases']]),dtype=int)]

        nudgesGroup = nudgesGroup.fillna(0).T

        # get stats across group of mice
        nudgesGroupMean = nudgesGroup.groupby(axis=1, level=0).mean()
        nudgesGroupStd = nudgesGroup.groupby(axis=1, level=0).std()
        nudgesGroupSem = nudgesGroup.groupby(axis=1, level=0).sem()
        nudgesGroupSum = nudgesGroup.groupby(axis=1, level=0).sum()

        timesGroupMean = timesGroup.groupby(level=0).mean()
        timesGroupStd = timesGroup.groupby(level=0).std()
        timesGroupSem = timesGroup.groupby(level=0).sem()

        groups = {
            "nudgesGroupMean": nudgesGroupMean,
            "nudgesGroupStd": nudgesGroupStd,
            "nudgesGroupSem": nudgesGroupSem,
            "nudgesGroupSum": nudgesGroupSum,
            "timesGroupMean": timesGroupMean,
            "timesGroupStd": timesGroupStd,
            "timesGroupSem": timesGroupSem
        }

        return groups


    def saveGroupWebData(self, files=None, directory=None, destfolder=()):
        print('start saveGroupebData')
        fnames = Utils().Getlistoffiles(files=files, directory=directory, scorer=scorerWeb)
        data = self.collateGroupWebData(files=fnames, directory=None)

        MeanFilenameNudges = "%s_NudgesMeanGroup.h5" % os.path.basename(destfolder)
        StdFilenameNudges = "%s_NudgesStdGroup.h5" % os.path.basename(destfolder)
        SemFilenameNudges = "%s_NudgesSemGroup.h5" % os.path.basename(destfolder)
        SumFilenameNudges = "%s_NudgesSumGroup.h5" % os.path.basename(destfolder)

        MeanFilenameTimes = "%s_TimesMeanGroup.h5" % os.path.basename(destfolder)
        StdFilenameTimes = "%s_TimesStdGroup.h5" % os.path.basename(destfolder)
        SemFilenameTimes = "%s_TimesSemGroup.h5" % os.path.basename(destfolder)

        # save group files as hdf in dest folder
        data["nudgesGroupMean"].to_hdf("%s\\%s" % (destfolder, MeanFilenameNudges), key='NudgesMean', mode='a')
        print("Group Dataframe with mean webcam nudge values file saved for %s" % os.path.basename(destfolder))
        data["nudgesGroupStd"].to_hdf("%s\\%s" % (destfolder, StdFilenameNudges), key='NudgesStd', mode='a')
        print("Group Dataframe with standard deviations of webcam nudges saved for %s" % os.path.basename(destfolder))
        data["nudgesGroupSem"].to_hdf("%s\\%s" % (destfolder, SemFilenameNudges), key='NudgesSem', mode='a')
        print("Group Dataframe with standard error of the mean of webcam nudges saved for %s" % os.path.basename(destfolder))
        data["nudgesGroupSum"].to_hdf("%s\\%s" % (destfolder, SumFilenameNudges), key='NudgesSum', mode='a')
        print("Group Dataframe with sums of webcam nudges saved for %s" % os.path.basename(destfolder))

        data["timesGroupMean"].to_hdf("%s\\%s" % (destfolder, MeanFilenameTimes), key='TimesMean', mode='a')
        print("Group Dataframe with mean webcam times values file saved for %s" % os.path.basename(destfolder))
        data["timesGroupStd"].to_hdf("%s\\%s" % (destfolder, StdFilenameTimes), key='TimesStd', mode='a')
        print("Group Dataframe with standard deviations of webcam times saved for %s" % os.path.basename(destfolder))
        data["timesGroupSem"].to_hdf("%s\\%s" % (destfolder, SemFilenameTimes), key='TimesSem', mode='a')
        print("Group Dataframe with standard error of the mean of webcam times saved for %s" % os.path.basename(destfolder))


    def getAllWebcamStuff(self, directory):
        # dir here is the analysis folder with all the directories for each date
        dirs = glob(os.path.join(directory, "*"))

        # i keep plots folder in this dir so delete this from list to iterate over
        dirs = [i for i in dirs if 'plots' not in i]
        dirs = [i for i in dirs if 'temporary' not in i]

        for l in range(0, len(dirs)):
            try:
                self.saveGroupWebData(directory=dirs[l], destfolder=dirs[l])
                print("Group data saved for %s" % dirs[l])
            except:
                print("couldn't run analysis for: %s" % dirs[l])
        print("Analysis finished!")



# plan: do time spent in quadrants (?), speed, make up graphs for webcam stuff, do time-related stuff for side cam data, step cycles


# # to find any backward runs: (runs is the dataframe, so would need to loop through multiples for batch processing)
# mask = np.logical_and.reduce((
#     runs.loc(axis=1)['Tail1','likelihood'] > 0.9,
#     runs.loc(axis=1)['Tail1','y'] < runs.loc(axis=1)['Nose','y'],
#     runs.loc(axis=1)['Nose','likelihood'] > 0.9,
#     runs.loc(axis=1)['Nose','y'] < runs.loc(axis=1)['Stage_R','y'],
#     runs.loc(axis=1)['Tail1','y'] < runs.loc(axis=1)['Stage_R','y'],
#     runs.loc(axis=1)['Stage_R','likelihood'] > 0.9,
#     runs.loc(axis=1)['Pen','likelihood'] < 0.9,
#     runs.loc(axis=1)['Nose','x'] < runs.loc(axis=1)['Stage_R','x'],
#     runs.loc(axis=1)['Stage_L','x'] < runs.loc(axis=1)['Nose','x']))