import numpy as np, pandas as pd
import functools as fn
import math
import sys
np.random.seed(3)
datalen = 1
rate = 1

# Read csv
df = pd.read_csv(sys.argv[1], sep=',', header='infer')
dfy = pd.read_csv(sys.argv[2], sep=',', header=None, names=['Ans'])

# Reshape

# Filter feature

def extract(df):
    #  df['PM2.5s2'] = x**0.15
    #  for n in ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']:
    org = pd.read_csv(sys.argv[1], sep=',', header='infer')
    for n in df.columns:
        x = org[n]
        df[n] = (df[n] - x.mean()) / x.std()
    df = df.sort_index(axis=1)
    return df
df = extract(df)
#  print df['RAINFALL'].max()
#  mean = np.float64(0.0)
#  std = np.float64(1.0)
#  mean = x.values.mean()
#  std = x.values.std()
#  df['PM2.5'] = ((x-x.mean())/x.std())
#  df['PM2.52'] = x ** 2

init_w = pd.DataFrame(
          np.random.randn(datalen, len(df.columns)),
          columns=df.columns)


df['b'] = pd.Series([1]*len(dfy['Ans']), index=dfy.index)
dataset = pd.concat([df, dfy], axis=1).values
init_w = np.append(init_w.values, [1, 0])

np.random.shuffle(dataset)

validate = dataset[:len(dataset)/9]
train = dataset[len(dataset)/9:]

print train[:10]
#  train = train[:5]
#  trainAns = trainAns[:5]
#  print mean
#  print std

def linearRegression(w, x):
    return np.matmul(x, w)

def logisticRegression(w, x):
    z = np.matmul(x, w)
    return 1 / ( 1 + np.exp(-z) )

def rmse(w, x):
    return ((estimate(w, x) ** 2).sum() / len(x)) ** 0.5

def catAccuracy(w, x):
    y = x[:,-1]
    e = np.round(estimate(w, x))
    return (e*y + (1-e)*(1-y)).sum()/len(x)
    

def crossEntropy(w, x):
    fx = estimate(w, x)
    y = x[:, -1]
    return (y*fx + (1-y)*(1-fx)).sum()

    
def rmseGrad(w, x, lam):
    err = estimate(w, x) - x[:, -1]
    n = len(x)
    first = ((err**2).sum()*n)**(-0.5)
    reg = lam * w
    reg[-2] = 0
    res = np.matmul(err, x)*first - reg
    res[-1] = 0
    return res

def crossEntropyGrad(w, x, lam):
    err = estimate(w, x) - x[:, -1]
    reg = lam * w
    reg[-2] = 0
    res = np.matmul(err, x) - reg
    res[-1] = 0
    return res

estimate = logisticRegression
score = catAccuracy
loss = crossEntropy
grad = crossEntropyGrad

earlyQuit = 9
def adagrad(iw, x, rate, lam, count):
    sd = np.zeros_like(iw)
    w = np.copy(iw)
    qr = 10000000
    qs = 0
    v = x[:len(x)/9]
    x = x[len(x)/9:]
    while qr > 6:
        ws = [w] * (earlyQuit - 1)
        sds = [sd] * (earlyQuit - 1)
        best_w = w
        best_s = score(w, v)
        qr = int(math.sqrt(count))
        qc = 0

        for i in range(count):
            if(i%qr == 0):
                s = score(w, v)
                print "    count %d, x : %f, v: %f" %(qs + i, score(w, x), s)
                if s >= best_s:
                    best_w = w
                    best_s = s
                    qc = 0
                elif qc == earlyQuit - 2:
                    w = ws[0]
                    sd = sds[0]
                    if i > earlyQuit*qr:
                        qs = qs + i - earlyQuit*qr
                        count = earlyQuit*qr
                    else:
                        count = i
                    print '  back to %d ~ %d' % ( qs, qs+count )
                    qc = 1000000000
                    break
                else:
                    qc += 1
                ws.append(w.copy())
                ws = ws[-earlyQuit:]
                sds.append(sd.copy())
                sds = sds[-earlyQuit:]
            #  g = grad(w, x, lam)
            g = grad(w, x[np.random.choice(x.shape[0], 500)], lam)
            sd += g ** 2
            sd[-1] = 1
            w -= rate * g / (sd**0.5)

        if qc <= earlyQuit - 2:
            qr = 0
    return best_w

#  best_score = float('inf')
best_score = 0
best_w = None
best_lr = None
best_lam = None
#  for elr in range(0,5):
for elr in range(4,5):
    lr = pow(10, -(elr/4.0))
    #  for elam in range(6, 10):
    for elam in range(4, 7):
        lam = pow(3, -(elam/10.0))
        print "rate: %f, lam: %.20f" %(lr, lam)
        w = adagrad(init_w, train, lr, lam, int(100*(4**elr)))
        val_score = score(w, validate)
        #  if val_score < best_score:
        if val_score > best_score:
            best_score = val_score
            best_w = w
            best_lr = lr
            best_lam = lam
        print "  rate: %f, lam: %.20f, train : %f, verify: %f" %(lr, lam, score(w, train), val_score)
        print w.mean()

print "Recalculate best w"
#  for i in range(5):
    #  np.random.shuffle(dataset)

    #  v = dataset[:len(dataset)/9]
    #  t = dataset[len(dataset)/9:]
    #  print "rate: %f, lam: %.20f" %(best_lr, best_lam)
    #  w = adagrad(init_w, t, best_lr, best_lam, 300000)
    #  print "  rate: %f, lam: %.20f, train : %f, verify: %f" %(best_lr, best_lam, score(w, t), score(w, v))

print "rate: %f, lam: %.20f" %(best_lr, best_lam)
best_w = adagrad(init_w, dataset, best_lr, best_lam, 300000)
print "  rate: %f, lam: %.20f, train : %f" %(best_lr, best_lam, score(best_w, dataset))
print best_w
print "Calculating result"

test = pd.read_csv(sys.argv[3], sep=',', header='infer')
test = extract(test)
test['b'] = pd.Series([1]*len(test[test.columns[0]]), index=test.index)
test['Ans'] = pd.Series([1]*len(test[test.columns[0]]), index=test.index)

print test.values[0]
e = np.round(estimate(best_w, test.values))
print e[:10]

result = zip(test.index, e)
with open(sys.argv[4], 'w') as f:
    f.write('id,label\n' + ''.join(['%d,%d\n'%(int(id)+1, int(val)) for id, val in result]))
