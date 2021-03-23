import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm

tqdm.pandas()

# datadir = "H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\DLC_DualBelt-Holly-2020-12-28\\analysed_videos"
# outputdir = "H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\analysis"
# video = "HM-20201130FL_cam0_1.avi"
# scorer = "DLC_resnet50_DLC_DualBeltDec28shuffle1_200000"
# os.chdir("H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\analysis")
# dataname = str(Path(video).stem) + scorer + '.h5'
# Dataframe = pd.read_hdf(os.path.join(dataname))
# Dataframe.head()

## Temporary file, later change to loop through my directories
DataframeCoor = pd.read_hdf(
    r"C:\Users\Holly Morley\Documents\Documents\Temp_files\HM-202012081034272FL_cam0_1DLC_resnet_50_DLC_DualBeltFeb11shuffle1_800000.h5")
DataframeSkel = pd.read_hdf(
    r"C:\Users\Holly Morley\Documents\Documents\Temp_files\HM-202012081034272FL_cam0_1DLC_resnet_50_DLC_DualBeltFeb11shuffle1_800000_skeleton.h5")

scorer = 'DLC_resnet_50_DLC_DualBeltFeb11shuffle1_800000'
RunIdxNose = np.zeros(shape=DataframeCoor.shape)
bodyparts = DataframeCoor.columns.get_level_values(1)  # you can read out the header to get body part names!
bodyparts2plot = bodyparts

'''
def getInFrame(Dataframe,bp,pcutoff=0.9):
    #scorer = Dataframe.columns.get_level_values(0)[0]
    present = np.logical_and.reduce(
                (Dataframe[scorer][bp]['likelihood'] > pcutoff,
                 Dataframe[scorer]['Nose']['likelihood'] > pcutoff,#means that not counted if nose disappears for whatever reason
                 Dataframe[scorer]['Nose']['x'] > 20,
                 Dataframe[scorer]['Nose']['x'] < 1900
                 ))
    return present


for bpindex, bp in enumerate(bodyparts2plot):  # doing this in a loop is very slow, try find better way
    print("Analysing body part number %d out of %d" % (bpindex, len(bodyparts2plot)))
    RunIdx[:, bpindex] = DataframeCoor.progress_apply(lambda x: getInFrame(x, bp, pcutoff=0.9), axis=1)
'''

'''
def getInFrame(Dataframe, pcutoff=0.9):
    # scorer = Dataframe.columns.get_level_values(0)[0]
    present = np.logical_and.reduce(
        (Dataframe[scorer]['Nose']['likelihood'] > pcutoff,
         # means that not counted if nose disappears for whatever reason
         Dataframe[scorer]['Nose']['x'] > 20,
         Dataframe[scorer]['Nose']['x'] < 1900
         ))
    return present

def getInFrame(Dataframe, pcutoff=0.9):
    # scorer = Dataframe.columns.get_level_values(0)[0]
    present = np.logical_and.reduce(
        (Dataframe[scorer, 'Nose', 'likelihood'] > pcutoff,
         # means that not counted if nose disappears for whatever reason
         Dataframe[scorer, 'Nose', 'x'] > 20,
         Dataframe[scorer, 'Nose', 'x'] < 1900
         ))

 '''


def getInFrame(Dataframe, pcutoff=0.9):
    # scorer = Dataframe.columns.get_level_values(0)[0]
    present = np.logical_and.reduce(
        (pd.DataFrame(Dataframe.loc(axis=1)['Nose_likelihood']) > pcutoff,
         # means that not counted if nose disappears for whatever reason
         pd.DataFrame(Dataframe.loc(axis=1)['Nose_x']) > 20,
         pd.DataFrame(Dataframe.loc(axis=1)['Nose_x']) < 1900
         ))
    return present


### Find values where 'Nose' is in frame (creates 1D boolean array)
FlatDataframeCoor = DataframeCoor.loc(axis=1)[scorer]
FlatDataframeCoor.columns = [f'{x}_{y}' for x, y in FlatDataframeCoor.columns]
RunIdxNose = FlatDataframeCoor.progress_apply(lambda x: getInFrame(x, pcutoff=0.9), axis=1)

### Filter original data by Nose index (RunIdxNose). All data where 'Nose' not in frame is chucked out.
ReducedDataframeCoor = DataframeCoor[RunIdxNose]

### Find the beginning and ends of runs and chunk them into their respective runs
# chunkIdxstart = ReducedDataframeCoor[scorer, 'Nose', 'x'].shift(1) - ReducedDataframeCoor[scorer, 'Nose', 'x'] > 1500  # finds frame where x coordinate of 'Nose' jumps from very high to low number
# chunkIdxend = ReducedDataframeCoor[scorer, 'Nose', 'x'].shift(-1) - ReducedDataframeCoor[scorer, 'Nose', 'x'] < -1500  # finds frame where before the x coordinate of 'Nose' jumps to a much lower number
chunkIdxstart = ReducedDataframeCoor['Nose_x'].shift(1) - ReducedDataframeCoor[
    'Nose_x'] > 1500  # finds frame where x coordinate of 'Nose' jumps from very high to low number
chunkIdxend = ReducedDataframeCoor['Nose_x'].shift(-1) - ReducedDataframeCoor[
    'Nose_x'] < -1500  # finds frame where before the x coordinate of 'Nose' jumps to a much lower number
chunks = pd.DataFrame({'Start': chunkIdxstart, 'End': chunkIdxend})  # puts both series into one dataframe
chunks['Start'].values[0] = True  # adds very first and very last values to dataframe
chunks['End'].values[-1] = True
frameNos = pd.DataFrame({'Start': chunks.index[chunks['Start']], 'End': chunks.index[
    chunks['End']]})  # creates dataframe of just the frame numbers for the starts and stops of runs
# StartIdx = chunks.index[chunks['Start']]
# EndIdx = chunks.index[chunks['End']]

numRuns = len(frameNos)
Run = np.zeros([len(ReducedDataframeCoor)])
ReducedDataframeCoor['Run'] = Run
FrameIdx = ReducedDataframeCoor.index
ReducedDataframeCoor['FrameIdx'] = FrameIdx
for i in range(0, len(frameNos)):
    mask = np.logical_and(ReducedDataframeCoor.index >= frameNos['Start'][i],
                          ReducedDataframeCoor.index <= frameNos['End'][i])
    ReducedDataframeCoor['Run'][mask] = i
ReducedDataframeCoor.set_index(['Run', 'FrameIdx'], append=False, inplace=True)

