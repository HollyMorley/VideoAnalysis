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
            (Dataframe.loc['Nose', 'likelihood'] > pcutoff,  # no need to specify rows as only one row fed into this function below
             # means that not counted if nose disappears for whatever reason
             Dataframe.loc['Nose', 'x'] > 20,
             Dataframe.loc['Nose', 'x'] < 1900
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
        chunkIdxstart = self.ReducedDataframeCoor.loc(axis=1)['Nose', 'x'].shift(1) - self.ReducedDataframeCoor.loc(axis=1)['Nose', 'x'] > 1500  # finds frame where x coordinate of 'Nose' jumps from very high to low number
        chunkIdxend = self.ReducedDataframeCoor.loc(axis=1)['Nose', 'x'].shift(-1) - self.ReducedDataframeCoor.loc(axis=1)['Nose', 'x'] < -1500  # finds frame where before the x coordinate of 'Nose' jumps to a much lower number
        chunks = pd.DataFrame({'Start': chunkIdxstart, 'End': chunkIdxend})  # puts both series into one dataframe
        chunks.loc(axis=1)['Start'].values[0] = True  # adds very first value to dataframe
        chunks.loc(axis=1)['End'].values[-1] = True  # adds very last value to dataframe
        frameNos = pd.DataFrame({'Start': chunks.index[chunks['Start']], 'End': chunks.index[chunks['End']]})  # creates dataframe of just the frame numbers for the starts and stops of runs


        Run = np.zeros([len(self.ReducedDataframeCoor)])
        self.ReducedDataframeCoor.loc(axis=1)['Run'] = Run
        FrameIdx = self.ReducedDataframeCoor.index
        self.ReducedDataframeCoor.loc(axis=1)['FrameIdx'] = FrameIdx
        for i in range(0, len(frameNos)):
            mask = np.logical_and(self.ReducedDataframeCoor.index >= frameNos.loc(axis=1)['Start'][i],
                                  self.ReducedDataframeCoor.index <= frameNos.loc(axis=1)['End'][i])
            #self.ReducedDataframeCoor.loc(axis=1)['Run'][mask] = i
            self.ReducedDataframeCoor.loc[mask, 'Run'] = i
        self.ReducedDataframeCoor.set_index(['Run', 'FrameIdx'], append=False, inplace=True)
        return self.ReducedDataframeCoor

    def Main(self, destfolder=(), files=None, directory=None, pcutoff=pcutoff, scorer=scorer):
        # if inputting file paths, make sure to put in [] even if just one
        # to get file names here, just input either files = [''] or directory = '' into function
        # destfolder should be raw format
        files = utils.Utils().Getlistoffiles(files, directory)
        for j in range(0, len(files)):
            DataframeCoor = pd.read_hdf(files[j])
            DataframeCoor = DataframeCoor.copy()
            print("starting analysis")
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