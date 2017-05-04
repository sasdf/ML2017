import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('historyDNN', 'r') as f:
    hist = pickle.load(f)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
tra, = ax1.plot(hist['tra'], color='#0000ff', label='train_accuracy')
vala, = ax1.plot(hist['vala'], color='#ff0000', label='validation_accuracy')
trl, = ax2.plot(hist['trl'], color='#2288bb', label='train_loss')
vall, = ax2.plot(hist['vall'], color='#bb2288', label='validation_loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
b1 = ax1.get_position()
b2 = ax2.get_position()
ax1.set_position([b1.x0, b1.y0+b1.height*0.2, b1.width, b1.height*0.8])
ax2.set_position([b2.x0, b2.y0+b2.height*0.2, b2.width, b2.height*0.8])

plt.legend(handles=[tra, vala, trl, vall], ncol=2, bbox_to_anchor=(0, -0.3, 1., .102), loc='lower center', borderaxespad=0., mode='expand')
plt.show()
