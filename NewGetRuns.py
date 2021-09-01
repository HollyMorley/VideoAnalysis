import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm
import utils
from Config import *
from pathlib import Path

tqdm.pandas()

# datadir = "H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\DLC_DualBelt-Holly-2020-12-28\\analysed_videos"
# outputdir = "H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\analysis"
# video = "HM-20201130FL_cam0_1.avi"
# scorer = "DLC_resnet50_DLC_DualBeltDec28shuffle1_200000"
# os.chdir("H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\analysis")
# dataname = str(Path(video).stem) + scorer + '.h5'
# Dataframe = pd.read_hdf(os.path.join(dataname))
# Dataframe.head()


class GetRuns:

    def __init__(self): # MouseData is input ie list of h5 files
        super().__init__()


    def getInFrame(self, Dataframe, pcutoff):
        # scorer = Dataframe.columns.get_level_values(0)[0]
        present = np.logical_and.reduce(
            (Dataframe.loc['Tail1', 'likelihood'] > 0.999,
             Dataframe.loc['Tail1', 'x'] > 5,
             Dataframe.loc['Tail1', 'x'] < 1915
             ))
        return present

    def filterData(self, DataframeCoor, pcutoff, scorer):
        ### Find values where 'Nose' is in frame (creates 1D boolean array)
        FlatDataframeCoor = DataframeCoor.loc(axis=1)[scorer]  ###THIS IS A COPY
        #FlatDataframeCoor.columns = [f'{x}_{y}' for x, y in FlatDataframeCoor.columns]
        RunIdxNose = FlatDataframeCoor.progress_apply(lambda x: self.getInFrame(x, pcutoff), axis=1)

        ### Filter original data by Nose index (RunIdxNose). All data where 'Nose' not in frame is chucked out.
        self.ReducedDataframeCoor = FlatDataframeCoor[RunIdxNose]
        self.ReducedDataframeCoor = self.ReducedDataframeCoor.copy()  # THIS IS NOT IDEAL BUT CANT FIND ANOTHER SOLUTION

        ### Find the beginning and ends of runs and chunk them into their respective runs
        errorMask = np.logical_and.reduce(( # creates a mask to chuck out any wrong tracking data. Logic is that if only tail1 is present it must be mislabelled
            self.ReducedDataframeCoor.loc(axis=1)['Platform', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Tail2', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Tail3', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['RForepaw', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['RHindpaw', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['RAnkle', 'likelihood'] < pcutoff,
        ))
        self.ReducedDataframeCoor = self.ReducedDataframeCoor.drop(self.ReducedDataframeCoor[errorMask].index)

        chunkIdxend = np.logical_or.reduce((
                                    # ideal scenario: end is when in the next frame the tail base is on the left hand side of the frame, the tail base is very far right in the frame and the nose is not visible
                                    np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(-1) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] < -1500, # finds frame before the x coordinate of 'Tail1' jumps to a much lower number
                                                        self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] > 1800,
                                                        self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] < pcutoff)),
                                    # alt scenario 1: end is when in the next frame the tail base is on the left hand side of the frame and the average likelihood of Nose, Shoulder, Hump and Hip jumps from low to high in the next frame. In other words, if the above logic with the tail fails, see in which frame do all the (front end) body points go from being not present to present
                                    np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(-1) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] < -1500, # finds frame before the x coordinate of 'Tail1' jumps to a much lower number
                                                        (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'].shift(-1) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'].shift(-1) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'].shift(-1) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'].shift(-1)) / 4 -
                                                        (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood']) / 4 > 0.75
                                                            )),
                                    # alt scenario 2: same as above logic but looking at two frames ahead instead of one
                                    np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(-2) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] < -1500, # finds frame before the x coordinate of 'Tail1' jumps to a much lower number
                                                        (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'].shift(-2) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'].shift(-2) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'].shift(-2) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'].shift(-2)) / 4 -
                                                        (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood']) / 4 > 0.75
                                                            ))
                                    ))

        chunkIdxstart = np.logical_or.reduce((np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(1) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] > 1500, # finds frame where x coordinate of 'Nose' jumps from very high to low number
                                                            self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(1) > 1800
                                                             )),
                                      np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(1) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] > 1500,
                                                          (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood']) / 4 -
                                                          (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'].shift(1) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'].shift(1) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'].shift(1) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'].shift(1)) / 4 > 0.75
                                                            )),
                                      np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(2) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] > 1500,
                                                          (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood']) / 4 -
                                                          (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'].shift(2) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'].shift(2) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'].shift(2) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'].shift(2)) / 4 > 0.75
                                                            ))
                                      ))

        chunks = pd.DataFrame({'Start': chunkIdxstart, 'End': chunkIdxend}, index= self.ReducedDataframeCoor.index)  # puts both series into one dataframe
        chunks.loc(axis=1)['Start'].values[0] = True  # adds very first value to dataframe
        chunks.loc(axis=1)['End'].values[-1] = True  # adds very last value to dataframe
        StartNos = chunks.index[chunks['Start']]
        StartNos = StartNos[abs(np.roll(StartNos, -1) - StartNos) > 1000]
        EndNos = chunks.index[chunks['End']]
        EndNos = EndNos[abs(np.roll(EndNos, -1) - EndNos) > 1000]

        # remove incorrect frames if any I am aware of:
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1822.9940185546875: # MR 30-11-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1883.6925048828125: # MR 01-12-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1883.5736083984375: # FL 03-12-2020
            StartNos = StartNos[0:-2]
            EndNos = EndNos[0:-2]
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1883.2921142578125: # FR 03-12-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1883.7017822265625: # MR 07-12-2020
            StartNos = StartNos.delete(25)
            EndNos = EndNos.delete(25)
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1887.3360595703125: # FR 11-12-2020
            StartNos = StartNos.delete(4)
            EndNos = EndNos.delete(4)
            StartNos = StartNos[0:-2]
            EndNos = EndNos[0:-2]
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1887.249755859375: # MR 11-12-2020
            StartNos = StartNos.delete([25,26])
            EndNos = EndNos.delete([25,26])
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1886.79248046875: #FL 14-12-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1901.6219482421875: # FLR 15-12-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1885.9906005859375: # FL 16-12-2020
            StartNos = StartNos.delete([11, 27])
            EndNos = EndNos.delete([11, 27])
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1883.7584228515625: # FR 16-12-2020
            StartNos = StartNos[0:-2]
            EndNos = EndNos[0:-2]
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1888.09228515625: # FLR 17-12-2020
            StartNos = StartNos.delete(25)
            EndNos = EndNos.delete(25)
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1888.1256103515625: # FR 17-12-2020
            StartNos = StartNos.delete(25)
            EndNos = EndNos.delete(25)
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1886.1680908203125: # FL 18-12-2020
            StartNos = StartNos.delete(25)
            EndNos = EndNos.delete(25)
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1886.28564453125: # FR 18-12-2020
            StartNos = StartNos.delete([25,26])
            EndNos = EndNos.delete([25,26])

        frameNos = pd.DataFrame({'Start': StartNos, 'End': EndNos})  # creates dataframe of just the frame numbers for the starts and stops of runs
        print('Number of runs detected: %s' % len(frameNos))

        #Run = np.zeros([len(self.ReducedDataframeCoor)])
        Run = np.full([len(self.ReducedDataframeCoor)], np.nan)
        self.ReducedDataframeCoor.loc(axis=1)['Run'] = Run
        FrameIdx = self.ReducedDataframeCoor.index
        self.ReducedDataframeCoor.loc(axis=1)['FrameIdx'] = FrameIdx
        for i in range(0, len(frameNos)):
            mask = np.logical_and(self.ReducedDataframeCoor.index >= frameNos.loc(axis=1)['Start'][i],
                                  self.ReducedDataframeCoor.index <= frameNos.loc(axis=1)['End'][i])
            #self.ReducedDataframeCoor.loc(axis=1)['Run'][mask] = i
            self.ReducedDataframeCoor.loc[mask, 'Run'] = i
        self.ReducedDataframeCoor = self.ReducedDataframeCoor[self.ReducedDataframeCoor.loc(axis=1)['Run'].notnull()]
        self.ReducedDataframeCoor.set_index(['Run', 'FrameIdx'], append=False, inplace=True)

        self.ReducedDataframeCoor = self.findRunStart(self.ReducedDataframeCoor)

        return self.ReducedDataframeCoor

    def findRunStart(self,data):
        startRALL = list()
        for r in range(0, len(data.index.unique(level='Run'))):
            mask = np.logical_and(data.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'] > pcutoff, data.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'] > pcutoff)
            if sum(data.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'x'][mask] < data.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'x'][mask]) > 0:
                print('backwards runs for run: %s' % r)
                lastbackwards = np.where(np.logical_and.reduce((data.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'x'] < data.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'x'], data.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'] > pcutoff,data.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'] > pcutoff)))[0][-1]
                try:
                    for i in reversed(range(0, len(data.xs(r, axis=0, level='Run')))):
                        if np.logical_or.reduce((
                            np.logical_and.reduce((
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Nose', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Nose', 'x'] > data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'x'],
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'x'] < 960,
                                i > lastbackwards,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Nose', 'likelihood'] < pcutoff
                            )),
                            np.logical_and.reduce((
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Nose', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Nose', 'x'] > data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'x'],
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'x'] < 960,
                                i > lastbackwards,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Tail1', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Tail2', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Tail2', 'x'] > data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Tail1', 'x']
                            )),
                            np.logical_and.reduce((
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Nose', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Nose', 'x'] > data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'x'],
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'x'] < 960,
                                i > lastbackwards,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Tail1', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Tail3', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Tail3', 'x'] > data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Tail1', 'x']
                            ))
                        )):
                            startR = data.xs(r, axis=0, level='Run').iloc(axis=0)[i].name
                            time = (startR/330)/60
                            mins = int(time)
                            secs = (time - mins)*60
                            print('Real run starts at %s minutes, %s seconds' % (mins, secs))
                            startRALL.append(startR)
                            break

                except(NameError):
                    print("mouse run blocked")
                    for i in reversed(range(0, len(data.xs(r, axis=0, level='Run')))):
                        if np.logical_and.reduce((
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Nose', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'likelihood'] > pcutoff,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Nose', 'x'] > data.xs(r, axis=0, level='Run').iloc(axis=0)[i].loc['Tail1', 'x'],
                                i > lastbackwards,
                                data.xs(r, axis=0, level='Run').iloc(axis=0)[i-1].loc['Nose', 'likelihood'] < pcutoff)):
                                    startR = data.xs(r, axis=0, level='Run').iloc(axis=0)[i].name
                                    time = (startR / 330) / 60
                                    mins = int(time)
                                    secs = (time - mins) * 60
                                    print('Real run starts at %s minutes, %s seconds' % (mins, secs))
                                    startRALL.append(startR)
                                    break
            else:
                print('no backwards runs for run: %s' % r)
                startR = data.xs(r, axis=0, level='Run').iloc(axis=0)[0].name
                startRALL.append(startR)

        if len(startRALL) == len(data.index.unique(level='Run')):
            startMask = data.index.get_level_values(level='FrameIdx').isin(startRALL)
            data.loc[startMask, 'RunStart'] = 1
        else:
            print('something went wrong')

        # get last frame number of each run
        endALL = list()
        for r in range(0, len(data.index.unique(level='Run'))):
            end = data.loc(axis=0)[r].tail(1).index[0]
            endALL.append(end)

        # chuck out all frames before the run where the mouse actually steps across
        dropmaskALL = list()
        for r in range(0, len(data.index.unique(level='Run'))):
            dropmask = np.logical_and(data.loc(axis=0)[r].index >= startRALL[r], data.loc(axis=0)[r].index <= endALL[r])
            dropmaskALL.append(dropmask)
        newdropmaskALL = list(np.concatenate(dropmaskALL).flat)
        todrop = np.invert(newdropmaskALL)
        data.drop(data.index[todrop], inplace=True)

        return data

    def Main(self, destfolder=(), files=None, directory=None, pcutoff=pcutoff, scorer=scorer):
        # if inputting file paths, make sure to put in [] even if just one
        # to get file names here, just input either files = [''] or directory = '' into function
        # destfolder should be raw format
        # if '20201218' in directory:
        #     files = utils.Utils().Getlistoffiles(files, directory, diffscorer)
        # else:
        #     files = utils.Utils().Getlistoffiles(files, directory)
        files = utils.Utils().Getlistoffiles(files, directory,scorer=scorer)
        for j in range(0, len(files)):
            DataframeCoor = pd.read_hdf(files[j])
            DataframeCoor = DataframeCoor.copy()
            print("starting analysis")
            ##### for 18-12-2020 files, give 'diffscorer' instead
            if '20201218' in files[j]:
                self.filterData(DataframeCoor, pcutoff, diffscorer)
            else:
                self.filterData(DataframeCoor, pcutoff, scorer)

            # save reduced dataframe as a .h5 file for each mouse
            destfolder = destfolder
            newfilename = "%s_RunsAll.h5" %Path(files[j]).stem
            self.ReducedDataframeCoor.to_hdf("%s\\%s" %(destfolder, newfilename), key='RunsAll', mode='a')
            print("Reduced coordinate file saved for %s" % files[j])
            del DataframeCoor

        print("Finished extracting runs for files: \n %s" %files)

    def getOriginalDF(self, files=None, directory=None, pcutoff=pcutoff, scorer=scorer):
        files = utils.Utils().Getlistoffiles(files, directory)
        for j in range(0, len(files)):
            DataframeCoor = pd.read_hdf(files[j])
            DataframeCoor = DataframeCoor.copy()
            DataframeCoor = DataframeCoor.loc(axis=1)[scorer]
        return DataframeCoor


## to use this:
## >> GetRuns().Main(destfolder= r"FOLDER_PATH", directory = r"PATH")
## or
## >> GetRuns().Main(destfolder= r"FOLDER_PATH", files = [r"PATH", r"PATH", ...])
##
## or if from python terminal:
## GetRuns.GetRuns().Main(destfolder= r"FOLDER_PATH", directory = r"PATH")