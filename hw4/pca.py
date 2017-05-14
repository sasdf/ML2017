import string, numpy as np, matplotlib.pyplot as plt
from PIL import Image

subs = string.ascii_uppercase[:10]
nums = list(range(10))
fn = ['pcadb/%s%02d.bmp' % (s,n) for s in subs for n in nums]
images = [ Image.open(n) for n in fn ]
data = np.array([np.array(im).flatten() for im in images])
data = data.astype('float64') / 255.0
for im in images:
    im.close()
mean = data.mean(axis=0)
datac = data - mean
u, s, v = np.linalg.svd(datac)

for i in range(100):
    reco = datac @ v[:i].T @ v[:i]
    err = ((datac - reco)**2).mean()**0.5
    #  print ("%d: %f" % (i, err))
    if err < 0.01:
        print ("Smallest k: %d" % i)
        break;

reco = datac @ v[:5].T @ v[:5] + mean

fig = plt.figure()
fig.suptitle('Average face')
plt.imshow(mean.reshape(64,64),cmap='gray')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.show()

fig = plt.figure()
fig.suptitle('Top 9 eigenfaces')
for i in range(9):
    ax = fig.add_subplot(3, 3, i+1)
    ax.imshow(v[i].reshape(64,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('{:.3f}'.format(s[i]))
plt.show()

fig = plt.figure()
fig.suptitle('Original 100 faces')
for i in range(100):
    ax = fig.add_subplot(10, 10, i+1)
    ax.imshow(data[i].reshape(64,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
plt.show()

fig = plt.figure()
fig.suptitle('Reconstructed 100 faces')
for i in range(100):
    ax = fig.add_subplot(10, 10, i+1)
    ax.imshow(reco[i].reshape(64,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
plt.show()


