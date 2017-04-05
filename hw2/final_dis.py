import numpy as np, pandas as pd
import functools as fn
import math
import sys
import pickle
# Filter feature
def transform(df):
    for n in ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']:
        pass
        df[n+'**0.125'] = (df[n]-df[n].min())**0.125
        df[n+'**0.25'] = (df[n]-df[n].min())**0.25
        df[n+'**0.5'] = (df[n]-df[n].min())**0.5
        df[n+'log'] = (df[n]-df[n].min()+1).map(math.log)
        for p in range(2,20):
            df[n+'**'+str(p)] = df[n]**p
    #  df['age**2'] = df['capital_gain']**2
    return df
def extract(df):
    #  df['PM2.5s2'] = x**0.15
    df = transform(df)
    org = transform(pd.read_csv(sys.argv[1], sep=',', header='infer'))
    for n in df.columns:
        x = org[n]
        df[n] = (df[n] - x.mean()) / x.std()
    for n in ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']:
        pass
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
        g = grad(w, x[np.random.choice(x.shape[0], 500)], lam)
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

earlyQuit = 256
seed = 6
np.random.seed(seed)
datalen = 1
rate = 1

# Read csv
#  df = pd.read_csv(sys.argv[1], sep=',', header='infer')
#  dfy = pd.read_csv(sys.argv[2], sep=',', header=None, names=['Ans'])

#  # Reshape

#  df = extract(df)
#  #  print df['RAINFALL'].max()
#  #  mean = np.float64(0.0)
#  #  std = np.float64(1.0)
#  #  mean = x.values.mean()
#  #  std = x.values.std()
#  #  df['PM2.5'] = ((x-x.mean())/x.std())
#  #  df['PM2.52'] = x ** 2

#  init_w = pd.DataFrame(
          #  np.random.randn(datalen, len(df.columns)),
          #  columns=df.columns)


#  df['b'] = pd.Series([1]*len(dfy['Ans']), index=dfy.index)
#  dataset = pd.concat([df, dfy], axis=1).values
#  init_w = np.append(init_w.values, [1, 0])

#  np.random.shuffle(dataset)

#  validate = dataset[:len(dataset)/9]
#  train = dataset[len(dataset)/9:]

#  print train[:10]
#  #  train = train[:5]
#  #  trainAns = trainAns[:5]
#  #  print mean
#  #  print std


#  #  best_score = float('inf')
#  best_score = 0
#  best_w = None
#  best_lr = None
#  best_lam = None
#  for elr in range(3,4):
#  #  for elr in range(4,5):
    #  lr = pow(10, -(elr/2.0))
    #  #  for elam in range(0, 10):
    #  for elam in range(9, 10):
        #  lam = pow(3, -(elam/1.5))
        #  print "rate: %f, lam: %.20f" %(lr, lam)
        #  w = gradDesc(init_w, train, lr, lam, int(1000))
        #  #  w = gradDesc(init_w, train, lr, lam, int(100000))
        #  val_score = score(w, validate)
        #  #  if val_score < best_score:
        #  if val_score > best_score:
            #  best_score = val_score
            #  best_w = w
            #  best_lr = lr
            #  best_lam = lam
        #  print "  rate: %f, lam: %.20f, train : %f, verify: %f" %(lr, lam, score(w, train), val_score)
        #  print w.mean()

#  print "Recalculate best w"
#  print "rate: %f, lam: %.20f" %(best_lr, best_lam)
#  best_w = gradDesc(init_w, dataset, best_lr, best_lam, 100000)
#  print "  rate: %f, lam: %.20f, train : %f" %(best_lr, best_lam, score(best_w, dataset))

#  print "Verify"
#  #  for i in range(5):
    #  #  np.random.shuffle(dataset)

    #  #  v = dataset[:len(dataset)/9]
    #  #  t = dataset[len(dataset)/9:]
    #  #  print "rate: %f, lam: %.20f" %(best_lr, best_lam)
    #  #  w = gradDesc(init_w, t, best_lr, best_lam, 100000000)
    #  #  print "  rate: %f, lam: %.20f, train : %f, verify: %f" %(best_lr, best_lam, score(w, t), score(w, v))

#  print "best w"
#  print "  rate: %f, lam: %.20f, train : %f" %(best_lr, best_lam, score(best_w, dataset))
best_w = pickle.loads("""
Y251bXB5LmNvcmUubXVsdGlhcnJheQpfcmVjb25zdHJ1Y3QKcDAKKGNudW1weQpuZGFycmF5CnAx
CihJMAp0cDIKUydiJwpwMwp0cDQKUnA1CihJMQooSTIxOAp0cDYKY251bXB5CmR0eXBlCnA3CihT
J2Y4JwpwOApJMApJMQp0cDkKUnAxMAooSTMKUyc8JwpwMTEKTk5OSS0xCkktMQpJMAp0cDEyCmJJ
MDAKUydceGJiXHg4YVx4ZWFceGM2XHg5YVx4YzVceDlhP1xuXHg5NGZceDk3XHgwMVx4MWZceDlj
P0F+V3YzIlx4YjM/XHhjN1x4ZjVUQFx4YWFceGQzXHhhN1x4YmZVS1x4ZmVceGNmVFx4Y2JceDli
XHhiZkwvXHg4OFx4ZjJceGQzXHRceDkyXHhiZm5HXHhjMUljXHg4M0I/XHgwZlx4OTNDTlx4YTJk
XHhiMj98XHhlZWZceGExOFdceGMzP1wneGcpXHgxZFx4OWJceGE1XHhiZlx4YTFceGNiXHg4N2dj
XHhmMVx4ZGE/WVx4MGJTXHhjY2NceDFiXHhjZT9ceGQyXHhhOVx4OThceGU2XHhlYlx4YmRceGQy
P1x4ZTEiXHhkZlx4MDBceGY2XHhlMVx4ZTc/LFx4YjVceGJldFx4YzYqXHhlMz85cVx4OWJWXHgx
ZFwnXHhiZT9ceGU5XHhjYVx4ZmNceGVjWztceGJkP0NceGQyXHgxZFx4ZTFceDlkXHhiOFx4YWU/
XHhiZW9ceGJmXHgwN1x4YjZceDgzXHhhND9ceDE1XHhlODc4XHhmNCJceGFmP1x4ZGVDXHgxMlx4
OTJceGJiXHhlNFx4YmU/aT5ceDg4Olx4YWFceDAzXHhiYT9ceGFhXG5ceDBmKWp2XHhkNT9AXHhl
OFx4YThceDlmXHg5MVJceDg2P1x4ZmFybVx4ZWNlb1x4YjI/XFxgflx4ZTBnXHg5N1x4YjY/XHg4
YVx4MWFceGY0XVx4ZjJqXHhjMD9ceDlmXHhjN1JceGFkWFx4ZjBceGQzP0lceGJhXHhjMFx4ZTNF
XHhmNVx4YmNceGJmY1NceGY3elx4YzZceGM4XHhjMj9ceGMyXHhmOVx4Y2JceDk0XHhjMFx4YjBc
eGI0P1x4MGNceGMyXHg5NVx4OGZceDkzXHhjOFx4YzM/XHgwNlx4OWRceGRiSDR+XHg4OT9ceGJi
XHhiM1x4ZThceDEzXHhkY1x4OGVceGIyP1x4MWRceGExXFxceDkxXHhkZlxcXHhkYj9ceDgzXHhm
MFx4YzVceGZlXHgxMFx4YWNceGI2P1x4MWRKXHhhYWI2XHg3Zlx4YmRceGJmbFx4YmFceGQ1XHhk
Zlx4ODdceDlhXHhiM1x4YmZceGI1XHhkYk9EXHhjMT5ceDhhXHhiZk5ceGEwSn5sXHg5MFx4YTQ/
PGxceGJkXHgxOUtceDAzXHhhNj9ceGE5IVx4ZTZceDEzXHhmZVx4ZDdceGFkP1x4OTRceDAzLFx4
ZGNceDA0XHg5NVx4YjU/XHhiZG4yXHgwN1x4YWZwXHhhZD9RYUsmXHgxN1BceGIyP1x4ODdceDFm
XHhmOVx4ODdoXHgwNVx4YzE/XHhhYlx4YzRGXHgxMlx4YmNhXHhiOT9ceGM2I1x4ODMwRilceGI4
P1x4YzZceGNjdlx4Y2FceGRkW1x4OGU/P1x4ZDZceDFkXHg5MTFceDBlXHhhMD9ceDE1LVx4YTRI
XHhhNHZceDlhXHhiZlx4OTYgdVx4YzEldlx4YjU/XHg5N1x4ZjVceGY1XHhiMWFceGYyXHhmND9D
R1x4MTVceDE1XHg4Y1x4OGNceGIyP1x4ZTZ8eFx4ZDNceGZkT1x4ZTA/XHg4YVx4MTlceGZmXHg5
OVx4YzZcJ1x4Yzk/WFx4YThdPW9ceGI5XHhhYj9ceGI0XHg5Ylx4MTBdUn5ceGNkXHhiZlx4YTdc
eDAwXHhhNlx4ODBceGE0IVx4OTM/L1x4YzJceDFhXHhjMFx4MTZpXHhkNT8yIFx4ZDlceGZlOFx4
YTFceGM0P1x4YjVceDAwXHhjNVx4ZGRceGQ4dlx4YTFceGJmTlx4MGVceGY0XHgxNEdceDg2XHhj
Nlx4YmZceDA2VXZceGNjXHg4OVx4MWNceGNhXHhiZm9ceGM4XHhkN1x4OTN9YVx4YjVceGJmXHhm
YVx4MDVceGExJD9ceGMyXHhhMT9ceGMzXHhkOFx4Y2VceDllL1x4ZDJceGM2PytceDE3XHhhOGhc
eDAwXHhhMlx4YjY/eVx4YmVceDE0XHg4Nlx4ZTZxXHhiMT9ceGQzXHhiNlx4ZDJceGY5Ilx4ZTlc
eGQ5XHhiZlx4ZmZceDA0L1x4OWVceGNiXHhhMlx4ZTFceGJmXHhiZlx4YzFceGZkXHgwMF5ceGY1
XHhjMj9ceDk5XHhiOWxceGMyZ3ZceGQ4P0FceGNlKWFcciFceGNiP2VceGQzdFx4OGNRXHgwZVx4
YzI/XHhkYUpceDlkRUdceGE4XHhiMT9ceGMyXHg4ZGApJFxyXHhjMz9ceDgwXHhhMVx4YTVceGY1
XHhkZFx4ZGRceGEzP1x4YTZceGMyfCxceDE4XHhjM1x4Yjg/XHhkM1x4N2ZvXFxceDhiXHg5M1x4
OTZceGJmOEpceGQ0XHRqTVx4OGM/XHg5N1BceGVlXHgxOXxpXHhlMD9MR1x4OTJiXHg5MFx4ZDdc
eGE3P1x4MDVceGU3XHhjZVx4Y2ZceDgyXHg4M1x4OTI/XHg4NFx4ZjkqXHg4Nn53XHhiMz9ceGMw
Kl1ceGRmY1x4MWJceGMxPzNceGY0dlx4YWRcclZceDhkP1x4OGJceDg3XHgwMVxuIFx4Y2FceDkz
P1x4MThwXHhiZTRceDlhXHhhMVx4YTM/XHRceDA3XHgwY1x4ZGM5XHgxNlx4ZTQ/XHgwMkVEQ2Rc
eDE3XHhjMz9ceDg0XHg4OC1ceDk5XHg5YVx4YzlceGEwP1x4YzBceDhhXHhkNW5LXHg5Mlx4ZTk/
XHhhNChceDlkXHgxNFx4ODVAXHhjMD9VXHhkZVx4MWFceGZjelx4MTRceGQ0P1x4MDVceGE5XG5c
eGFkKiFceGQxXHhiZlx4MDFceDFlXHhiNFx4MWVceGUxXHhjNlx4YWQ/ZFZcJ1x4YmI6XHgxOVx4
Y2U/XHhhNFVceDFjXHhhYlx4YTdceDg4XHhjNz9ceDE3ZVx4OTRKa1x4ZjBceGMzXHhiZlx4MDJc
eDFiXHg4M1x4MTlceDkzXHg4YVx4ZTlceGJmXHg4MVx4OTRoXHhhNlxyLFx4ZTM/XHg4M3FXXHRc
eDE1Z1x4ZTc/elx4MTdceGZjXHgwMjp0XHhjMlx4YmZceGEwXHhiN1x4ZDRceGU0XHg5M1VceGEx
P1x4ODZceDlmXHg4OVx4ODlceGE3XHhmYlx4OGM/djRVXHhiYlx4OWZceDlhYj9ffVx4OGV3XHhk
M1x4OThceGFkP1x4MGZceGFjXHgxM1x4YWZrO1x4OGVceGJmNk0jXHhkM0RceGVmXHhhOFx4YmZV
SFx4MGVTXHgxMFx4ZjVceDhjXHhiZmtceGI3XHg5NFx4OTFceDFiXHhiZXZceGJmXHhiY1x4OTdc
eGMxXHhmM1x4ZGJceGY2dFx4YmZceGY5XHgxY2pceGRkXHhjMVx4ZWRceDhlXHhiZj9ceGRiXHhi
OVx4ODRceGRkXHhjNFx4ZDg/XHg5Ylx4YzRceGQyXHg5ZFx4ZTBceGZmXHhmMFx4YmYjXHhmN1x4
ZTJceDkzMlx4MTdceGZhP1x4ZDhcXFx4OGZceDg1Slx4ZjVceGVmXHhiZlx4MThceGFjVFx4MGN3
XHhhZVx4ZTBceGJmXHg5MyxkfmdAXHgxND80XHgxMFx4MWZceDFlJjNceGQzP1x4MDJVUVx4YmRB
cFx4YzM/W1x4ZTJbfEc4XHhlZD9ceGZlP1x4YjBceDk5XHhmY1x4OGVceDE5QDRceGJjXHhkN1x4
ZTJceDljXHg5Nlx4MDhceGMwXHhjMT9ceGU3XHg4NFx4MDBceDE5XHhlMj9ceGVmXHg4NVx4YzRh
XHhkNlx4ZjRceDEyQFx4YTdceGM1Tlx4YmEra1x4YjJceGJmXHgwMFx4ZTksXHhlY1x4MWRaXHhi
MVx4YmZAPlx4ZDBcclx4ODJceGFmXHhlMD9ceGVjXHgxZVx4OWJ1XHhlYVx4ZGVceGM3PzhceDhj
QWxceGQzNFx4ZDE/T2ZceDFlXHhmZlx4MTNceGIxXHhhZlx4YmZpR1x4ZTZjXHgwMlx4YTRceGNl
XHhiZmhceGZmXHhjY1x4ODdceDk0XHhkN1x4Y2JceGJmXHhjYVJceDFiIihKXHhiOT9ceGIwdTdc
eGY4XHgxMlx4YjRceGE0XHhiZjlceGZjSjxceDg4XHhmYlx4ZGFceGJmXHg5NFx4YzVceGU1XHhi
N1x4ZmNceDFkXHgxMEBceDgzXHgwNFx4YmFceDA1XHgxZlx4ZGMlXHhjMFx4MGNceGFiXHhjN1x4
MTJceGNjXHg5Mlx4YjVceGJmXHhmMFx4ZTIkXHgwMCZcXFx4YzM/TVx4ZGFceGRkXHhlMEJceGY2
XHhiYVx4YmZceGExXHhiM1x4ODhceGJmXHg5M1x4ZmZceGRiXHhiZlx4ZWM1alx4ZjZceGJiXHhi
Nlx4YzVceGJmcWlceGFmXHhkN1x4OGJceGQxXHhmYlx4YmZceDEwXHgxOF1ceGJmXHhjMlx4OGRc
eGYxXHhiZlx4YzdFXHg4YVx4ZmNceGU3XHg4OFx4YzVceGJmSiRceDAySlx4ZDJuXHhlNT9ceDgx
XHg5Zlx4YzY6XHg4Nlx4ZmFceGVjXHhiZjZceGNiXHgxZUZceGZmXHhmY1x4YjY/XHgxMlx4ZjVc
eGQwXHhiZWlsXHhiMVx4YmZceGE5NFx4MTgiXHhiNFx4YTNceDhiP2BceGM5XHhlY1o3XHhlZVx4
YmY/XHhmOV1ceGQ4XHhmNi1ceDk2XHhiNlx4YmYuXHhmMlx4MTcoXHgwNVx4OTdceDlmXHhiZlx4
MTdWLEhceDhic1x4YjJceGJmXHgwM2FceGMxXHhmYylceDkzXHhjZj93O1x4MTFoXHg5Zlx4MWVc
eGQzP0JceDBlXHhkMFl9XHhkYVx4YzNceGJmT3RceDkzXHhiMTNtXHhmNT83XHhjNnxceGQxXHhm
ZC9ceGU1P1x4OTBfXCdceGFmXHhjNHdceGI1XHhiZlx4OTVceDAyPFpceGI3X1x4ZTNceGJmXHhl
YVx4ZDlceGYyXHg4OFx4ZWFIXHhjYlx4YmZceGVlXHhhOVx4MDNceGYzXHgwNVx4YWJceGJkP1x4
YmZceDlmLUJDbFx4OThceGJmYlx4MTk/XHg4ZHNpXHhiZj9ceGYxXHhhY1x4ZWZceDA4XHhhNUdc
eDkyXHhiZlx4ZDlceDE2XCdceDE5XHhiZF5ceGM2XHhiZlx4OTFceDliXHgwNSxceDAxZVx4ZjVc
eGJmX1x4YmQ0dVx4Y2NJXHhlOT9ceDk4bTtceDgyJlx4MWVceGUxP1x4ODdceGI0XHhjMis8XHgx
YXRceGJmXHhmOS0+XHg4Mlx4OWRceDFjXHg3Zlx4YmZceGUwXHhjYlx4YzJceGI1IFx4ZThbP1x4
ODdceGQ4XHgxNFM3UFx4ODVceGJmXHg5Y1x4ODZceGE4MFx4ZDJceGUyXHg5ZFx4YmYrXHhmOFx4
YTFceDBlN2FceDk4P1x4MTlceGI5bFx4MTdFXHhkN3A/XHg5Nlx4ZTJceGUwXG5ceDBlXFxceDhh
P1x4ZDBwXHhhOVJceDgyXHhhOVx4ODY/XHgwMlx4ZGRceGVmK1x4ZmNceDFjTVx4YmZceGY2XHhl
NFx4ZGFoXHg4M01ceGM5XHhiZlx4OTVceGJkXHgxNVx4ZTU6XHg4Nlx4YmI/XHhkN1x4YTNUXHhj
OEZceDg0XHg5ND9ceDlhZlx4ZTZcJ2lceDgyXHg4ZT8ofGxceGY0XHg5OFx4ZjZceDlmP1x4YmJc
eGQ3UlJceDk4XHhkYVx4OTJceGJmIlx4ZmJceDE5XHg4Nlx4ZDZiXHg4MVx4YmZceDg2XHhkZW5c
eGY2XHg5Y1x4OTZceDg4P1x4ZTdceGU4NVpceGQ5XHhmNFx4ZDc/XHhhMUpceGU3XHg4Y1x4OGRc
eGI1XHhmNj9ceGQ3YCtceGM0XXhceGE3P1x4ODhcblx4MDI8XHhhMVx4YWRceGM5P1x4ZTdceDA3
Ylx4MDVceGZjZ1x4ZDk/XHhmMVx4YzVceDAzS1wnXHhmYVx4YjBceGJmXHhlM1x4MWFceGY2XHg4
YVx4ZThlXHg5ZFx4YmZceGVkR1x4OTFceGY4Zlx4OWVceGI3XHhiZlx4ZTkjXHhmN1x4ZTJceDk0
XHhlNFx4OWZceGJmXHhlYlx4YWNceGMwXHhhOVx4MTRceGRmXHg4Yj9ceGM3XHg5YVx4ZjRceGIw
XHhhYVx4YWNceGI1XHhiZlx4ZGZceDhiXHhmNVx4YzVceDFiXHgxOFx4OWQ/XHgwNnBceGQ1XHhk
MFx4ZTdceDkyXHhiZlx4YmZceDFlVFx4YzJmXHhlY1x4Y2ZceGMzP1x4ZDJMXHg5N3VrXHg5MFx4
YzFceGJmXHhiZlx4MGI4XHg5ZVx4OTdceGY3XHhmN1x4YmZRX1x4ZDVceDg4XHgwZVx4MDVceGYw
PyhceDkxXHgxN1x4YjJceGJkXHhkYVx4Yzg/SlxyXHhlMlx4YjdceDhjXHhkM1x4ZDVceGJmXHhl
ZVx4MDdceDllcnZuXHhmNFx4YmZceGNhXHhlNlx4ZWIyS1x4YzJceGQ1PzJceGJhXHg5YWZceGI5
a1x4ZTM/QWhceGFiXHhlNEZceDFmXHg5NT9ceGIye1x4ZDdceGU3XHhhNFx4YmNceGViXHhiZndc
eGRlXHg4ZFx4YmVceGVjXHg4NFx4ZDk/WFx4ZDJjXHhjYlx4YTRAXHgwNFx4YzBceDAwXHgwMFx4
MDBceDAwXHgwMFx4MDBceDAwXHgwMCcKcDEzCnRwMTQKYi4=""".decode('base64'))
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

print pickle.dumps(best_w).encode('base64')

