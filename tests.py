import matplotlib.pyplot as plt
import numpy as np

z = np.array([[0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.]])
# z.itemsize()
print(z.sum(0))
# zHist = [[a + a1], [b + b1], [c + c1], [d + d1] for a, b, c, d in range(0, len(z))]
zHist = z.sum(0)
zz = ['clap', 'hiHat', 'kick', 'snare']

plt.bar(zz, height=zHist)
# plt.hist(zHist)
plt.show()