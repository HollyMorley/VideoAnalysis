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

class GetWebcamRuns:
    def __init__(self): # MouseData is input ie list of h5 files
        super().__init__()

    def findRuns(self, Dataframe, pcutoff=pcutoffWeb):
        FlatDataframe = Dataframe.loc(axis=1)[scorerWeb]
        if FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 698.312255859375:
            cutstart = 3000
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 345.91314697265625:
            cutstart = 2500
        else:
            cutstart = FlatDataframe.iloc(axis=0)[np.where(FlatDataframe.loc(axis=1)['Door_L', 'likelihood'] > 0.6)[0][0]].name

        chunkStart = np.logical_or.reduce((np.logical_and.reduce(
            (FlatDataframe.loc(axis=1)['Stage_L', 'likelihood'] > pcutoff,
             FlatDataframe.loc(axis=1)['Stage_L', 'likelihood'].shift(-1) > pcutoff,
             FlatDataframe.loc(axis=1)['Door_L', 'likelihood'] < pcutoff,
             FlatDataframe.index > cutstart
             )), np.logical_and.reduce(
            (FlatDataframe.loc(axis=1)['Stage_L', 'likelihood'] > pcutoff,
             FlatDataframe.loc(axis=1)['Stage_L', 'likelihood'].shift(-1) > pcutoff,
             FlatDataframe.loc(axis=1)['Door_R', 'likelihood'] < pcutoff,
             FlatDataframe.index > cutstart
             )), np.logical_and.reduce(
            (FlatDataframe.loc(axis=1)['Stage_R', 'likelihood'] > pcutoff,
             FlatDataframe.loc(axis=1)['Stage_R', 'likelihood'].shift(-1) > pcutoff,
             FlatDataframe.loc(axis=1)['Door_L', 'likelihood'] < pcutoff,
             FlatDataframe.index > cutstart
             )), np.logical_and.reduce(
            (FlatDataframe.loc(axis=1)['Stage_R', 'likelihood'] > pcutoff,
             FlatDataframe.loc(axis=1)['Stage_R', 'likelihood'].shift(-1) > pcutoff,
             FlatDataframe.loc(axis=1)['Door_R', 'likelihood'] < pcutoff,
             FlatDataframe.index > cutstart
             )))
        )

        Platform_Lmask = FlatDataframe.loc(axis=1)['Platform_L', 'y'] > pcutoff
        Platform_Rmask = FlatDataframe.loc(axis=1)['Platform_R', 'y'] > pcutoff
        meanPlatform_L = np.mean(FlatDataframe.loc(axis=1)['Platform_L', 'y'][Platform_Lmask])
        meanPlatform_LX = np.mean(FlatDataframe.loc(axis=1)['Platform_L', 'x'][Platform_Lmask])
        meanPlatform_RX = np.mean(FlatDataframe.loc(axis=1)['Platform_R', 'x'][Platform_Rmask])
        midmouse = (FlatDataframe.loc(axis=1)['Tail1', 'y'] + FlatDataframe.loc(axis=1)['Nose', 'y'])/2
        halfmouse = abs(FlatDataframe.loc(axis=1)['Tail1', 'y'] - FlatDataframe.loc(axis=1)['Nose', 'y'])/2
        extraatend = 50

        chunkEnd = np.logical_or(np.logical_and.reduce(
            (#FlatDataframe.loc(axis=1)['Platform_L', 'likelihood'] > pcutoff,
             FlatDataframe.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
             FlatDataframe.loc(axis=1)['Tail1', 'likelihood'] > pcutoff,
             #FlatDataframe.loc(axis=1)['Nose', 'y'] < FlatDataframe.loc(axis=1)['Platform_L', 'y'],
             #FlatDataframe.loc(axis=1)['Tail1', 'y'] < FlatDataframe.loc(axis=1)['Platform_L', 'y'],
             #midmouse < meanPlatform_L - (halfmouse + extraatend),
             midmouse < meanPlatform_L - (halfmouse),
             FlatDataframe.loc(axis=1)['Door_L', 'likelihood'] < 0.9,
             FlatDataframe.loc(axis=1)['Door_R', 'likelihood'] < 0.9,
             FlatDataframe.index > cutstart,
             FlatDataframe.loc(axis=1)['Nose', 'x'] > meanPlatform_LX + 12,
             FlatDataframe.loc(axis=1)['Nose', 'x'] < meanPlatform_RX - 12
             )), np.logical_and.reduce(
            (#FlatDataframe.loc(axis=1)['Platform_L', 'likelihood'] > pcutoff,
             FlatDataframe.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
             FlatDataframe.loc(axis=1)['Tail1', 'likelihood'] > pcutoff,
             #midmouse < meanPlatform_L - (halfmouse + extraatend),
             midmouse < meanPlatform_L - (halfmouse),
             #FlatDataframe.loc(axis=1)['Door_L', 'y'] < 700,
             FlatDataframe.loc(axis=1)['Door_R', 'y'] < 700,
             FlatDataframe.index > cutstart,
             FlatDataframe.loc(axis=1)['Nose', 'x'] > meanPlatform_LX + 12,
             FlatDataframe.loc(axis=1)['Nose', 'x'] < meanPlatform_RX - 12))
        )

        both = chunkStart.astype(int) + chunkEnd.astype(int)  # creates single array, where 0 = not run, 1 = run has started, 2 = run has finished (can be 1 after 2, THIS IS NOT RUN)
        #start = np.logical_and(both == 1, np.roll(both, 1) == 0, np.roll(both, -15) == 1)
        start = np.logical_and.reduce((both == 1, np.roll(both, 1) == 0, np.roll(both, -4) == 1, np.roll(both, 2) == 0,
                                       np.roll(both, 4) == 0,
                                       np.roll(both, 8) == 0,
                                       np.roll(both, 13) == 0,
                                       np.roll(both, 14) == 0, np.roll(both, 38) == 0, np.roll(both, 39) == 0))
        #end = np.logical_and.reduce((both == 2, np.roll(both, 1) <= 1, np.roll(both, 2) <= 1, np.roll(both, 3) <= 1, np.roll(both, 4) <= 1, np.roll(both,5) <= 1, np.roll(both,10) <= 1, np.roll(both,15) <= 1, np.roll(both,25) <= 1, np.roll(both, -1) == 2))
        end = np.logical_and.reduce((both == 2, np.roll(both, 1) <= 1, np.roll(both, 2) <= 1, np.roll(both, 3) <= 1, np.roll(both, 4) <= 1, np.roll(both,5) <= 1, np.roll(both,6) <= 1, np.roll(both,10) <= 1, np.roll(both,15) <= 1, np.roll(both,25) <= 1))
        s = np.where(start)
        s = s[0]
        toocloseS = s - np.roll(s, 1)
        maskS = abs(toocloseS) > 200
        s = s[maskS]
        e = np.where(end)
        e = e[0]
        toocloseE = e - np.roll(e,1)
        maskE = abs(toocloseE) > 200
        e = e[maskE]

        if FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 852.6813354492188: # MR 20201201
            e = np.array([x for x in e if x < 14200])
            s = np.array([x for x in s if x < 14200])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 731.4928588867188: # FL 20201203
            e = np.array([x for x in e if x < 17700])
            s = np.array([x for x in s if x < 17700])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 1027.0849609375: # MR 20201203
            e = np.array([x for x in e if x > 9000 or x < 4000])
            s = np.array([x for x in s if x > 9000 or x < 4000])
            e = np.array([x for x in e if x > 12150 or x < 11800])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 550.6144409179688: # FR 20201203
            s = np.array([x for x in s if x > 9400 or x < 9000])
            e = np.array([x for x in e if x > 4100 or x < 3900])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 1082.9298095703125: # MR 20201204
            s = np.array([x for x in s if x > 8650 or x < 8400])
            s = np.array([x for x in s if x > 1900 or x < 1600])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 416.04754638671875: # FR 20201207
            s = np.array([x for x in s if x > 12600 or x < 12400])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 883.034423828125: # MR 20201207
            e = np.array([x for x in e if x < 13400 or x > 13700])
            s = np.array([x for x in s if x < 13400 or x > 13700])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 493.0966491699219: # FR 20201210
            e = np.array([x for x in e if x > 8500 or x < 7500])
            e = np.array([x for x in e if x < 21700])
            s = np.array([x for x in s if x < 21700])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 714.1786499023438: # FL 20201210
            e = np.array([x for x in e if x > 6500 or x < 4400])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 714.1727905273438: # FR 20201211
            e = np.array([x for x in e if x > 5900 or x < 4000])
            s = np.array([x for x in s if x > 5900 or x < 4000])
            e = np.array([x for x in e if x < 18500])
            s = np.array([x for x in s if x < 18500])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 714.2410888671875:  # MR 20201211
            e = np.array([x for x in e if x > 20000 or x < 18500])
            s = np.array([x for x in s if x > 20000 or x < 18500])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 874.5956420898438: # FL 20201214
            s = np.array([x for x in s if x < 18200])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 1094.027587890625: # FR 20201214
            e = np.array([x for x in e if x > 18800 or x < 18500])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 828.329833984375:  # FLR 20201215
            e = np.array([x for x in e if x < 18200])
            s = np.array([x for x in s if x < 18200])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 535.1986083984375: # MR 20201215
            e = np.array([x for x in e if x > 22900 or x < 22400])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 6.531143665313721: # FL 20201215
            e = np.array([x for x in e if x > 19600 or x < 19300])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] ==  732.0282592773438: # FL 20201216
            e = np.array([x for x in e if x > 8600 or x < 8300])
            s = np.array([x for x in s if x > 8600 or x < 8300])
            e = np.array([x for x in e if x > 3060 or x < 2800])
            e = np.array([x for x in e if x > 16750 or x < 16450])
            s = np.array([x for x in s if x > 16750 or x < 16450])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 292.213623046875:  # FR 20201216
            e = np.array([x for x in e if x < 21400])
            s = np.array([x for x in s if x < 21400])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 707.5443725585938:  # FLR 20201217
            e = np.array([x for x in e if x > 12220 or x < 11910])
            s = np.array([x for x in s if x > 12220 or x < 11910])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 702.7410278320312:  # FR 20201217
            e = np.array([x for x in e if x > 16570 or x < 16080])
            s = np.array([x for x in s if x > 16570 or x < 16080])
            e = np.array([x for x in e if x > 15600 or x < 15300])
            s = np.array([x for x in s if x > 15600 or x < 15300])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 1091.2864990234375:  # FL 20201218
            e = np.array([x for x in e if x > 15130 or x < 14800])
            s = np.array([x for x in s if x > 15130 or x < 14800])
            s = np.array([x for x in s if x > 2000])
        elif FlatDataframe.loc(axis=1)['Tail3', 'y'].iloc[-1] == 1084.2406005859375:  # FR 20201218
            e = np.array([x for x in e if x > 24770 or x < 22990])
            s = np.array([x for x in s if x > 24770 or x < 22990])
            e = np.array([x for x in e if x > 21720 or x < 21430])




        if s.size == e.size:
            print("Runs correctly identified")
            # get data from each run using start and end frame indexes
            frameNos = pd.DataFrame({'Start': s, 'End': e})  # puts both arrays into one dataframe

            # find all frames between start and finish frames and label by their run number
            FlatDataframe = FlatDataframe.copy()
            for i in range(0, len(frameNos)):
                mask = np.logical_and(FlatDataframe.index >= frameNos.loc(axis=1)['Start'][i],
                                      FlatDataframe.index <= frameNos.loc(axis=1)['End'][i])
                # self.ReducedDataframeCoor.loc(axis=1)['Run'][mask] = i
                FlatDataframe.loc[mask, 'Run'] = i

            FilteredDataframe = FlatDataframe[FlatDataframe['Run'].notna()] # chuck out frames not in a run
            FilteredDataframe.set_index('Run', append =True, inplace=True)

            FilteredDataframe = FilteredDataframe.copy()
            FilteredDataframe.loc[:, 'TrialStart'] = np.nan
            FilteredDataframe.loc[:, 'RunStart'] = np.nan
            FilteredDataframe.loc[:, 'RunEnd'] = np.nan
            FilteredDataframe.loc[:, 'Quadrants'] = np.nan

            tsmask = FilteredDataframe.index.get_level_values(level=0).isin(s)
            emask = FilteredDataframe.index.get_level_values(level=0).isin(e)

            FilteredDataframe.loc[tsmask, 'TrialStart'] = 1
            FilteredDataframe.loc[emask, 'RunEnd'] = 1


            # find Start of Run frame (this is the last time the mouse crosses the stage boundary in a run)
            #attempt 2: do this by finding the closest frame to e where Tail < StageR (tracking on run tends to be better and less variable)
            startRALL = list()
            for r in range(0, len(frameNos)):
                StageRmask = FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_R', 'likelihood'] > pcutoff
                StageRmean = np.mean(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_R', 'y'][StageRmask])
                mousemask = np.logical_and(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                                           FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'] > pcutoff)
                avhalfmouse = np.mean(abs(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'y'][mousemask] - FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'y'][mousemask])/2)
                stagemask = np.logical_and(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_L', 'likelihood'] > pcutoff,
                                           FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_R', 'likelihood'] > pcutoff)
                avmiddlestage = np.mean(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_L', 'y'][stagemask] - abs(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_L', 'y'][stagemask] - FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_R', 'y'][stagemask])/2)
                for i in reversed(range(0, len(FilteredDataframe.xs(r, axis=0, level=1)))):
                    middlemouse = (FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Nose', 'y'] + FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Tail1', 'y'])/2
                    middlestage = FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Stage_L', 'y'] - abs(FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Stage_L', 'y'] - StageRmean)/2

                    # if mouse is fully visible, find the last frame where middle coordinate of the mouse has not yet crossed the threshold of the middle of the stage
                    # OR if the mouse is occluded eg by pen, instead find the last frame where the middle coordinate of the mouse has not yet crossed the threshold of the middle of the stage by extimating from the tail1 (which is less likely to be occluded by pen here)
                    if np.logical_or(np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Nose', 'likelihood'] > pcutoff,
                                                            FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Tail1', 'likelihood'] > pcutoff,
                                                            FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Stage_L', 'likelihood'] > pcutoff,
                                                            middlemouse >= middlestage
                                                            )), np.logical_and.reduce((
                                                            FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Tail1', 'y'] - avhalfmouse > avmiddlestage,
                                                            FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].loc['Nose', 'likelihood'] < pcutoff
                                                            ))

                                     ):
                        startR = FilteredDataframe.xs(r, axis=0, level=1).iloc(axis=0)[i].name
                        startRALL.append(startR)
                        break

            if len(startRALL) == s.size:
                smask = FilteredDataframe.index.get_level_values(level=0).isin(startRALL)
                FilteredDataframe.loc[smask, 'RunStart'] = 1

                # Also make column with quadrant markers
                q1maskALL = list()
                q2maskALL = list()
                q3maskALL = list()
                q4maskALL = list()

                for r in range(0, len(frameNos)):
                    #calculate these values for each run just in case camera moved
                    #topY = np.mean([np.mean(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Platform_L', 'y']).astype(int),
                                    #np.mean(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Platform_R', 'y']).astype(int)])
                    platmask = FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Platform_L', 'likelihood'] > pcutoff
                    stagemask = np.logical_and(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_L', 'likelihood'] > pcutoff, FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_R', 'likelihood'] > pcutoff)
                    topY = np.mean(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Platform_L', 'y'][platmask]).astype(int)
                    halfStage = np.mean((FilteredDataframe.xs(r, axis=0, level=1).loc(axis=1)['Stage_L', 'y'][stagemask] - FilteredDataframe.xs(r, axis=0, level=1).loc(axis=1)['Stage_R', 'y'][stagemask])/2)
                    botY = np.mean(FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Stage_L', 'y'][stagemask] - halfStage)
                    full = abs(topY - botY)
                    quarter = full / 4
                    quad1 = botY
                    quad2 = botY - quarter
                    quad3 = botY - quarter * 2
                    quad4 = topY + quarter
                    nextNoseY =     FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'y'].shift(-1)
                    lastNoseY =     FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'y'].shift(1)
                    nextTail1Y  =   FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'y'].shift(-1)
                    lastTail1Y =    FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'y'].shift(1)
                    interpolatedMiddleMouse = (((lastNoseY + lastTail1Y)/2) + ((nextNoseY + nextTail1Y)/2))/2
                    middleMouse = (FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'y'] + FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'y'])/2
                    q1mask = np.logical_or(np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'] > pcutoff,
                                                                middleMouse <= quad1,
                                                                middleMouse > quad2,
                                                                np.array(FilteredDataframe.xs(r, axis=0, level='Run').index) >= np.array(FilteredDataframe.xs(r, axis=0, level='Run').index[FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['RunStart'] == 1])
                                                                )),
                                           np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'].shift(1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc( axis=1)['Nose', 'likelihood'].shift(-1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'].shift(1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'].shift(-1) > pcutoff,
                                                                interpolatedMiddleMouse <= quad1,
                                                                interpolatedMiddleMouse > quad2,
                                                                np.array(FilteredDataframe.xs(r, axis=0, level='Run').index) >= np.array(FilteredDataframe.xs(r, axis=0, level='Run').index[FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['RunStart'] == 1])
                                                                ))
                    )
                    q1maskALL.append(q1mask)
                    q2mask = np.logical_or(np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'] > pcutoff,
                                                                middleMouse <= quad2,
                                                                middleMouse > quad3,
                                                                np.array(FilteredDataframe.xs(r, axis=0, level='Run').index) >= np.array(FilteredDataframe.xs(r, axis=0, level='Run').index[FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['RunStart'] == 1])
                                                                )),
                                            np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'].shift(1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc( axis=1)['Nose', 'likelihood'].shift(-1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'].shift(1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'].shift(-1) > pcutoff,
                                                                interpolatedMiddleMouse <= quad2,
                                                                interpolatedMiddleMouse > quad3,
                                                                np.array(FilteredDataframe.xs(r, axis=0, level='Run').index) >= np.array(FilteredDataframe.xs(r, axis=0, level='Run').index[FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['RunStart'] == 1])
                                                                ))
                    )
                    q2maskALL.append(q2mask)
                    q3mask = np.logical_or(np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'] > pcutoff,
                                                                middleMouse <= quad3,
                                                                middleMouse > quad4,
                                                                np.array(FilteredDataframe.xs(r, axis=0, level='Run').index) >= np.array(FilteredDataframe.xs(r, axis=0, level='Run').index[FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['RunStart'] == 1])
                                                                )),
                                            np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'].shift(1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc( axis=1)['Nose', 'likelihood'].shift(-1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'].shift(1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'].shift(-1) > pcutoff,
                                                                interpolatedMiddleMouse <= quad3,
                                                                interpolatedMiddleMouse > quad4,
                                                                np.array(FilteredDataframe.xs(r, axis=0, level='Run').index) >= np.array(FilteredDataframe.xs(r, axis=0, level='Run').index[FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['RunStart'] == 1])
                                                                ))
                    )
                    q3maskALL.append(q3mask)
                    q4mask = np.logical_or(np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'] > pcutoff,
                                                                middleMouse <= quad4,
                                                                middleMouse > topY,
                                                                np.array(FilteredDataframe.xs(r, axis=0, level='Run').index) >= np.array(FilteredDataframe.xs(r, axis=0, level='Run').index[FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['RunStart'] == 1])
                                                                )),
                                            np.logical_and.reduce((FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Nose', 'likelihood'].shift(1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc( axis=1)['Nose', 'likelihood'].shift(-1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'].shift(1) > pcutoff,
                                                                FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['Tail1', 'likelihood'].shift(-1) > pcutoff,
                                                                interpolatedMiddleMouse <= quad4,
                                                                interpolatedMiddleMouse > topY,
                                                                np.array(FilteredDataframe.xs(r, axis=0, level='Run').index) >= np.array(FilteredDataframe.xs(r, axis=0, level='Run').index[FilteredDataframe.xs(r, axis=0, level='Run').loc(axis=1)['RunStart'] == 1])
                                                                ))
                    )
                    q4maskALL.append(q4mask)

                q1maskALL = list(np.concatenate(q1maskALL).flat)
                q2maskALL = list(np.concatenate(q2maskALL).flat)
                q3maskALL = list(np.concatenate(q3maskALL).flat)
                q4maskALL = list(np.concatenate(q4maskALL).flat)

                FilteredDataframe.loc[q1maskALL, 'Quadrants'] = 1
                FilteredDataframe.loc[q2maskALL, 'Quadrants'] = 2
                FilteredDataframe.loc[q3maskALL, 'Quadrants'] = 3
                FilteredDataframe.loc[q4maskALL, 'Quadrants'] = 4


            else:
                print("Error in 'Start of Run' identification")

        else:
            print("Error in run identification. Check data!")

        return FilteredDataframe

        ######### pcutoff works for end @ 0.6 #############

