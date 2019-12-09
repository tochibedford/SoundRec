import os
import librosa


dirWalk = './audio'

dirList = []
fileList = []
X = []


for root, dirs, files in os.walk(dirWalk):
    for dirr in dirs:
        dirList.append(dirr)


for root, dirs, files in os.walk(dirWalk):
    if len(files) != 0:
        fileList.append(files)
    else:
        continue

# vary time
def timeVary(audd, auddOut):
    timeStep = 0.81
    for i in range(0, len(dirList)):
        for j in fileList[i]:
            audd = "./audio/" + dirList[i] + "/" + j
            y, sr = librosa.load(audd, duration=2.97)
            for k in range(0, 28):
                if timeStep != 1.00:
                    auddOut = "./aug/" + dirList[i] + "/aug_stretch_" + str(timeStep)[:4] + "_" + j
                    y_stretched = librosa.effects.time_stretch(y, timeStep)
                    librosa.output.write_wav(auddOut, y_stretched, sr)
                    timeStep += 0.01
                    print(auddOut)
                else:
                    timeStep += 0.01
                    print('at 1')
                    continue
            timeStep = 0.81


def pitchVary():
    n_steps = [2.5, 1, -1.5, -2.5]
    for i in range(0, len(dirList)):
        for j in fileList[i]:
            audd = "./audio/" + dirList[i] + "/" + j
            y, sr = librosa.load(audd, duration=2.97)
            for k in n_steps:
                auddOut = "./aug_pitch/" + dirList[i] + "/aug_pitch_" + str(k) + "_" + j
                y_pitched = librosa.effects.pitch_shift(y, sr, k)
                librosa.output.write_wav(auddOut, y_pitched, sr)
                print(auddOut)
# pitchVary()