# datadir = "H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\DLC_DualBelt-Holly-2020-12-28\\analysed_videos"
# outputdir = "H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\analysis"
# video = "HM-20201130FL_cam0_1.avi"
# scorer = "DLC_resnet50_DLC_DualBeltDec28shuffle1_200000"
# os.chdir("H:\\APA_Project\\Data\\Behaviour\\Dual-belt_APAs\\analysis\\DLC_DualBelt\\DualBelt_Side\\analysis")
# dataname = str(Path(video).stem) + scorer + '.h5'
# Dataframe = pd.read_hdf(os.path.join(dataname))
# Dataframe.head()
import numpy as np

scorer = "DLC_resnet_50_DLC_DualBeltFeb11shuffle1_800000"
diffscorer = 'DLC_resnet50_DLC_DualBeltFeb11shuffle1_800000'
scorerWeb = 'DLC_resnet50_DLC_DualBelt_webcamApr22shuffle1_800000'
pcutoff = 0.9
pcutoffWeb = 0.6
webcamWrongMultiple = 2
webcamFPS = 30
sideFPS = 330
sideViewBeltLength = 30
LongSideBeltLength = 35
ShortSideBeltLength = 33
midBeltLength = (LongSideBeltLength + ShortSideBeltLength)/2
LongSideBeltLengthQuad = LongSideBeltLength/4
ShortSideBeltLengthQuad = ShortSideBeltLength/4
midBeltLengthQuad = midBeltLength/4
