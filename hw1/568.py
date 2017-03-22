import numpy as np, pandas as pd
import functools as fn
import math
np.random.seed(5)
datalen = 9
rate = 1

# Read csv
df = pd.read_csv('train.csv', sep=',', header='infer').replace('NR', 0)
df.columns = ['Date', 'Site', 'Type'] + df.columns[3:].tolist()

# Reshape
df = df.drop('Site', axis='columns')

df = pd.melt(df, id_vars=['Date', 'Type'])

df['Date'] = pd.to_datetime(df['Date']).astype(int)
df['Date'] = pd.to_datetime(
                 df['Date'] +
                 df['variable'].astype(int)*1e+9*3600)

df = df.drop('variable', axis='columns').pivot('Type', 'Date').value.T.astype(float)

# Filter feature
#  df = df[df['PM2.5'] >= 0]

def extract(df):
    df['Ans'] = np.copy(df['PM2.5'].values)
    df = df[['PM2.5', 'Ans']]
    #  df = df[['PM2.5', 'Ans', 'SO2', 'RAINFALL', 'O3']]
    x = df['PM2.5']
    #  df['PM2.5l'] = (x+1000000).map(math.log)
    #  df['O3'] = (df['O3']+2).map(math.log)
    #  df['SO2'] = (df['SO2']+2).map(math.log)
    #  df['CH4'] = (df['CH4']+2).map(math.log)
    #  df['RAINFALL'] = 1.01**df['RAINFALL']

    #  df['PM2.5s2'] = x**0.15
    df = df.sort_index(axis=1)
    return df
ddd = df
df = extract(df)
#  print df['RAINFALL'].max()
mean = np.float64(0.0)
std = np.float64(1.0)
#  mean = x.values.mean()
#  std = x.values.std()
#  df['PM2.5'] = ((x-x.mean())/x.std())
#  df['PM2.52'] = x ** 2

dd = df.drop('Ans', axis='columns')
kernel = pd.DataFrame(
          np.random.randn(datalen, len(dd.columns)),
          columns=dd.columns)

dataset = np.array([   
              np.append(
                  m.drop('Ans', axis='columns')[t:t+datalen].values.flatten(),
                  [m['Ans'][t+datalen], 1])
              for _, m in df.groupby(pd.TimeGrouper(freq='M'))
              for t in range(len(m) - datalen)
              if all(m['Ans'][t:t+datalen+1] >= 0)
          ]) 
#  print train[:10]

kernel = np.append(kernel.values.flatten(), [-1 ,1])

np.random.shuffle(dataset)

validate = dataset[:len(dataset)/9]
train = dataset[len(dataset)/9:]

#  train = train[:5]
#  trainAns = trainAns[:5]
#  print mean
#  print std

def rmse(ker, tra):
    return ((np.matmul(tra, ker) ** 2).sum() / len(tra)) ** 0.5
    
def grad(ker, tra, lam):
    err = np.matmul(tra, ker)
    n = len(tra)
    first = ((err**2).sum()*n)**(-0.5)
    reg = lam * ker
    reg[-1] = 0
    res = np.matmul(err, tra)*first - reg
    res[-2] = 0
    return res

def adagrad(ker, tra, rate, lam, count):
    sd = np.zeros_like(ker)
    ker = np.copy(ker)
    for i in range(count):
        g = grad(ker, tra, lam)
        #  g = grad(ker, tra[np.random.choice(tra.shape[0], 100)], lam)
        sd += g ** 2
        sd[-2] = 1
        ker -= rate * g / (sd**0.5)
    return ker

best_loss = float('inf')
best_ker = None
best_lr = None
best_lam = None
for elr in range(1,2):
    lr = pow(10, -elr)
    for elam in range(0, 20):
        lam = pow(10, -elam)
        ker = adagrad(kernel, train, lr, lam, 1000*(elr+1))
        val_loss = rmse(ker, validate)
        if val_loss < best_loss:
            best_loss = val_loss
            best_ker = ker
            best_lr = lr
            best_lam = lam
        print "rate: %f, lam: %.20f, train : %f, verify: %f" %(lr, lam, rmse(ker, train), val_loss)

print "Recalculate best kernel"
for i in range(5):
    np.random.shuffle(dataset)

    v = dataset[:len(dataset)/9]
    t = dataset[len(dataset)/9:]
    ker = adagrad(kernel, t, best_lr, best_lam, 12000)
    print "rate: %f, lam: %.20f, train : %f, verify: %f" %(best_lr, best_lam, rmse(ker, t), rmse(ker, v))

best_ker = adagrad(kernel, dataset, best_lr, best_lam, 20000)
print "rate: %f, lam: %.20f, train : %f" %(best_lr, best_lam, rmse(best_ker, dataset))
print best_ker
print "Calculating result"

test = pd.read_csv('t2.csv', sep=',', names=['id', 'Type'] + range(9)).replace('NR', 0)
testId = [ id for id, x in test.groupby('id') ]
test = np.array([
          np.append(
              extract(
                  x.drop('id', axis=1)
                      .set_index('Type').T
                      .sort_index(axis=1)
                      .astype(float)
              ).drop('Ans', axis=1)
              .values.flatten(),
              [0, 1])
          for id, x in test.groupby('id') ])
result = zip(testId, np.matmul(test, best_ker))
result.sort(key=lambda x: int(x[0][3:]))
with open('opt', 'w') as f:
    f.write('id,value\n' + ''.join(['%s,%f\n'%x for x in result]))
