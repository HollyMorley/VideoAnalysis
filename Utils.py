# Several functions to aid analysis e.g. define a single run

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tables

#datadir = "H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\DLC_DualBelt-Holly-2020-12-28\\analysed_videos"
#outputdir = "H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\analysis"
#video = "HM-20201130FL_cam0_1.avi"
#scorer = "DLC_resnet50_DLC_DualBeltDec28shuffle1_200000"
#os.chdir("H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\analysis")

#dataname = str(Path(video).stem) + scorer + '.h5'
#Dataframe = pd.read_hdf(os.path.join(dataname))
#Dataframe.head()
DataframeCoor = pd.read_hdf(r"C:\Users\Holly Morley\Documents\Documents\Temp_files\HM-202012081034272FL_cam0_1DLC_resnet_50_DLC_DualBeltFeb11shuffle1_800000.h5")
DataframeSkel = pd.read_hdf(r"C:\Users\Holly Morley\Documents\Documents\Temp_files\HM-202012081034272FL_cam0_1DLC_resnet_50_DLC_DualBeltFeb11shuffle1_800000_skeleton.h5")


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def Histogram(vector, color, bins):
    dvector = np.diff(vector)
    dvector = dvector[np.isfinite(dvector)]
    plt.hist(dvector, color=color, histtype='step', bins=bins)

print("hello")
# %%
def PlottingResults(Dataframe, bodyparts2plot, alphavalue=.2, pcutoff=.5, colormap='jet', fs=(4, 3)):
    ''' Plots poses vs time; pose x vs pose y; histogram of differences and likelihoods.'''
    plt.figure(figsize=fs)
    colors = get_cmap(len(bodyparts2plot), name=colormap)
    scorer = Dataframe.columns.get_level_values(0)[0]  # you can read out the header to get the scorer name!

    for bpindex, bp in enumerate(bodyparts2plot):
        Index = Dataframe[scorer][bp]['likelihood'].values > pcutoff  # creates an array of True/False values of whether likelihood value is bigger than set value, for each body part
        plt.plot(Dataframe[scorer][bp]['x'].values[Index], Dataframe[scorer][bp]['y'].values[Index], '.',
                 color=colors(bpindex), alpha=alphavalue)

    plt.gca().invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    # plt.savefig(os.path.join(tmpfolder,"trajectory"+suffix))
    plt.figure(figsize=fs)
    Time = np.arange(np.size(Dataframe[scorer][bodyparts2plot[0]]['x'].values))

    for bpindex, bp in enumerate(bodyparts2plot):
        Index = Dataframe[scorer][bp]['likelihood'].values > pcutoff
        plt.plot(Time[Index], Dataframe[scorer][bp]['x'].values[Index], '--', color=colors(bpindex), alpha=alphavalue)
        plt.plot(Time[Index], Dataframe[scorer][bp]['y'].values[Index], '-', color=colors(bpindex), alpha=alphavalue)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.xlabel('Frame index')
    plt.ylabel('X and y-position in pixels')
    # plt.savefig(os.path.join(tmpfolder,"plot"+suffix))

    plt.figure(figsize=fs)
    for bpindex, bp in enumerate(bodyparts2plot):
        Index = Dataframe[scorer][bp]['likelihood'].values > pcutoff
        plt.plot(Time, Dataframe[scorer][bp]['likelihood'].values, '-', color=colors(bpindex), alpha=alphavalue)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.xlabel('Frame index')
    plt.ylabel('likelihood')

    # plt.savefig(os.path.join(tmpfolder,"plot-likelihood"+suffix))

    plt.figure(figsize=fs)
    bins = np.linspace(0, np.amax(Dataframe.max()), 100)

    for bpindex, bp in enumerate(bodyparts2plot):
        Index = Dataframe[scorer][bp]['likelihood'].values < pcutoff
        X = Dataframe[scorer][bp]['x'].values
        X[Index] = np.nan
        Histogram(X, colors(bpindex), bins)
        Y = Dataframe[scorer][bp]['x'].values
        Y[Index] = np.nan
        Histogram(Y, colors(bpindex), bins)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.ylabel('Count')
    plt.xlabel('DeltaX and DeltaY')

    # plt.savefig(os.path.join(tmpfolder,"hist"+suffix))

def getInFrame(Dataframe,bp,pcutoff=0.9):
    #scorer = Dataframe.columns.get_level_values(0)[0]
    present = np.logical_and.reduce(
                (Dataframe[scorer][bp]['likelihood'] > pcutoff,
                 Dataframe[scorer]['Nose']['likelihood'] > pcutoff,#means that not counted if nose disappears for whatever reason
                 Dataframe[scorer]['Nose']['x'] > 20,
                 Dataframe[scorer]['Nose']['x'] < 1900
                 ))
    return present


def DefineSingleRunCoord(Dataframe, bodyparts2plot, alphavalue=.2, pcutoff=.9, colormap='jet', fs=(4, 3)):
    colors = get_cmap(len(bodyparts2plot), name=colormap)  ## NOT SURE NEED THIS HERE
    RunIdx = np.zeros(shape=Dataframe.shape)

    for bpindex, bp in enumerate(bodyparts2plot): #doing this in a loop is very slow, try find better way
        print("Analysing body part number %d out of %d" % (bpindex, len(bodyparts2plot)))
        RunIdx[:, bpindex] = Dataframe.progress_apply(lambda x: getInFrame(x, bp, pcutoff=0.9), axis=1)







bodyparts = DataframeCoor.columns.get_level_values(1)  # you can read out the header to get body part names!

bodyparts2plot = bodyparts  # you could also take a subset, i.e. =['snout']

#PlottingResults(Dataframe, bodyparts2plot, alphavalue=.2, pcutoff=.5, fs=(8, 4))
DefineSingleRunCoord(Dataframe,bodyparts2plot,alphavalue=.2, pcutoff=.5, fs=(8, 4))
