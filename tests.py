import time
import winsound

fi = open("./models/ImageClassV1.hdf5", "rb").read()
fi2 = open("./models/ImageClassVUnlocked.hdf5", "rb").read()
winsound.Beep(500, 100)
winsound.Beep(1000, 300)

print(fi)
for i in range(0, len(fi)):
    print(i)
    if bytes(fi[i]) == bytes(fi2[i]):
        print('same')
    else:
        print('different at line ', i+1)
        winsound.Beep(1000, 100)
        winsound.Beep(500, 300)
        time.sleep(10)
