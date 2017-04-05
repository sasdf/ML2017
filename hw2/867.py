import numpy as np, pandas as pd
import functools as fn
import math
import sys
seed = 6
np.random.seed(seed)
datalen = 1
rate = 1

# Read csv
df = pd.read_csv(sys.argv[1], sep=',', header='infer')
dfy = pd.read_csv(sys.argv[2], sep=',', header=None, names=['Ans'])

# Reshape

# Filter feature
def transform(df):
    for n in ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']:
        pass
        df[n+'**0.125'] = (df[n]-df[n].min())**0.125
        df[n+'**0.25'] = (df[n]-df[n].min())**0.25
        df[n+'**0.5'] = (df[n]-df[n].min())**0.5
        df[n+'log'] = (df[n]-df[n].min()+1).map(math.log)
        df[n+'**2'] = df[n]**2
        df[n+'**3'] = df[n]**3
        df[n+'**4'] = df[n]**4
        df[n+'**5'] = df[n]**5
        df[n+'**6'] = df[n]**6
        df[n+'**7'] = df[n]**7
        df[n+'**8'] = df[n]**8
        df[n+'**9'] = df[n]**9
        df[n+'**10'] = df[n]**10
        df[n+'**11'] = df[n]**11
        df[n+'**12'] = df[n]**12
        df[n+'**13'] = df[n]**13
        df[n+'**14'] = df[n]**14
    #  df['age**2'] = df['capital_gain']**2
    return df
def extract(df):
    #  df['PM2.5s2'] = x**0.15
    df = transform(df)
    org = transform(pd.read_csv(sys.argv[1], sep=',', header='infer'))
    for n in df.columns:
        x = org[n]
        df[n] = (df[n] - x.mean()) / x.std()
    #  for n in ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']:
        #  pass
        #  df[n+'**0.5'] = (df[n]-df[n].min())**0.5
        #  df[n+'log'] = (df[n]-df[n].min()+1).map(math.log)
        #  df[n+'**2'] = df[n]**2
        #  df[n+'**3'] = df[n]**3
        #  df[n+'**4'] = df[n]**4
        #  df[n+'**5'] = df[n]**5
        #  df[n+'**6'] = df[n]**6
        #  df[n+'**7'] = df[n]**7
        #  df[n+'**8'] = df[n]**8
        #  df[n+'**9'] = df[n]**9
    #  df = df.drop('age', axis=1)
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

earlyQuit = 256
def gradDesc(iw, x, rate, lam, count):
    qs = 0
    p = len(x)/10
    v = x[:p]
    v2 = x[p:2*p]
    x = x[2*p:]
    state = emptyState(iw, x, rate, lam, count)
    while True:
        states = [state] * earlyQuit
        best_score = score(state['w'], v)
        score_acc = best_score
        best_i = qs
        qr = int(math.sqrt(count)) / 2
        if qr == 0:
            qr == 1
        qc = 0

        for i in range(count/qr):
            states.append(state)
            states = states[-earlyQuit:]
            s = score(state['w'], v)
            score_acc = score_acc * 0.8 + s * 0.2
            if score_acc >= best_score:
                best_score = score_acc
                best_i = qs + i*qr
                qc = 0
                print "    count %d, x : %f, v: %f, a: %f *%d %f, v2: %f" %(qs + i*qr, score(state['w'], x), s, score_acc, qc, best_score, score(state['w'], v2))
            elif qc == earlyQuit - 2:
                print "    count %d, x : %f, v: %f, a: %f  %d %f, v2: %f" %(qs + i*qr, score(state['w'], x), s, score_acc, qc, best_score, score(state['w'], v2))
                break
            else:
                qc += 1
                print "    count %d, x : %f, v: %f, a: %f  %d %f, v2: %f" %(qs + i*qr, score(state['w'], x), s, score_acc, qc, best_score, score(state['w'], v2))
            state = optimizer(state, x, rate, lam, qr)

        if qc == 0 or qr < 2:
            break
        state = states[-qc-1]
        if i > qc:
            qs = best_i
            count = qc*qr
        else:
            count = i / 2
        print '  back to %d ~ %d' % ( qs, qs+count )
    return states[-qc-1]['w']

def adagradEmpty(w, x, rate, lam, count):
    return {
                'w': w.copy(),
                'sd': np.zeros_like(w),
                'seed': seed
           }

def adagrad(state, x, rate, lam, count):
    w = state['w'].copy()
    sd = state['sd'].copy()
    i_seed = state['seed']
    for i in range(count):
        np.random.seed(i_seed)
        #  g = grad(w, x, lam)
        g = grad(w, x[np.random.choice(x.shape[0], 10)], lam)
        sd += g ** 2
        sd[-1] = 1
        w -= rate * g / (sd**0.5)
        i_seed += 1
    return {'w': w, 'sd': sd, 'seed': i_seed}

def adamEmpty(w, x, rate, lam, count):
    return {
                'w': w.copy(),
                'm': np.zeros_like(w),
                'v': np.zeros_like(w),
                'b1t': 1,
                'b2t': 1,
                'seed': seed
           }

def adam(state, x, rate, lam, count):
    w = state['w'].copy()
    m = state['m'].copy()
    v = state['v'].copy()
    b1t = state['b1t']
    b2t = state['b2t']
    i_seed = state['seed']
    a = 0.001
    b1 = 0.9
    b2 = 0.999
    e = 10e-8
    for i in range(count):
        np.random.seed(i_seed)
        #  g = grad(w, x, lam)
        g = grad(w, x[np.random.choice(x.shape[0], 1000)], lam)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*(g**2)
        b1t *= b1
        b2t *= b2
        c = a * (m/(1-b1t)) / ((v/(1-b2t))**0.5 + e)
        c[-1] = 0
        w -= c
        i_seed += 1
    return {
                'w': w,
                'm': m,
                'v': v,
                'b1t': b1t,
                'b2t': b2t,
                'seed': i_seed
           }

estimate = logisticRegression
score = catAccuracy
loss = crossEntropy
grad = crossEntropyGrad
#  emptyState = adagradEmpty
#  optimizer = adagrad
emptyState = adamEmpty
optimizer = adam

#  best_score = float('inf')
best_score = 0
best_w = None
best_lr = None
best_lam = None
for elr in range(3,4):
#  for elr in range(4,5):
    lr = pow(10, -(elr/2.0))
    #  for elam in range(0, 10):
    for elam in range(9, 10):
        lam = pow(3, -(elam/1.5))
        print "rate: %f, lam: %.20f" %(lr, lam)
        w = gradDesc(init_w, train, lr, lam, int(1000))
        #  w = gradDesc(init_w, train, lr, lam, int(100000))
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
print "rate: %f, lam: %.20f" %(best_lr, best_lam)
best_w = gradDesc(init_w, dataset, best_lr, best_lam, 100000)
print "  rate: %f, lam: %.20f, train : %f" %(best_lr, best_lam, score(best_w, dataset))

print "Verify"
#  for i in range(5):
    #  np.random.shuffle(dataset)

    #  v = dataset[:len(dataset)/9]
    #  t = dataset[len(dataset)/9:]
    #  print "rate: %f, lam: %.20f" %(best_lr, best_lam)
    #  w = gradDesc(init_w, t, best_lr, best_lam, 100000000)
    #  print "  rate: %f, lam: %.20f, train : %f, verify: %f" %(best_lr, best_lam, score(w, t), score(w, v))

print "best w"
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
