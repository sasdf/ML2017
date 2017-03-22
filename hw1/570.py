import numpy as np, pandas as pd
import functools as fn
import math
np.random.seed(3)
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
    x = df['PM2.5']
    df['PM2.5l'] = (x+100000).map(math.log)
    df = df.sort_index(axis=1)
    return df

df = extract(df)
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

train = np.array([   
        [m.drop('Ans', axis='columns')[t:t+datalen].values.flatten(), m['Ans'][t+datalen]]
        for _, m in df.groupby(pd.TimeGrouper(freq='M'))
        for t in range(len(m) - datalen)
        if all(m['Ans'][t:t+datalen+1] >= 0)
    ]) 
#  print train[:10]

kernel = kernel.values.flatten()

np.random.shuffle(train)

validate = train[:len(train)/15]
validateAns = validate[:, 1]
validateAns = np.array(validateAns.tolist())
validate = np.array(validate[:, 0].tolist())
train = train[len(train)/15:]
trainAns = train[:, 1]
trainAns = np.array(trainAns.tolist())
train = np.array(train[:, 0].tolist())


# Add bias
validate = np.append(validate, [[1]]*len(validate), axis=1)
train = np.append(train, [[1]]*len(train), axis=1)
kernel = np.append(kernel, [1])

#  train = train[:5]
#  trainAns = trainAns[:5]
print mean
print std

def rmse(ker, tra, traa):
    return (((np.matmul(tra, ker) - traa) ** 2).sum() / len(tra)) ** 0.5
    
def grad(ker, tra, traa, lam):
    err = np.matmul(tra, ker) - traa
    n = len(tra)
    first = (((err**2).sum()/n)**(-0.5))/n
    reg = lam * ker
    reg[-1] = 0
    return np.matmul(err, tra)*first - reg

def adagrad(ker, tra, traa, rate, lam, count):
    sd = np.zeros_like(ker)
    ker = np.copy(ker)
    for i in range(count):
        g = grad(ker, train, trainAns, lam)
        sd += g ** 2
        ker -= rate * g / (sd.sum()**0.5)
    return ker

best_loss = float('inf')
best_ker = None
best_lr = None
best_lam = None
for elr in range(4):
    lr = pow(10, -elr)
    for elam in range(1, 7):
        lam = pow(10, -elam)
        ker = adagrad(kernel, train, trainAns, lr, lam, 4000)
        val_loss = rmse(ker, validate, validateAns)
        if val_loss < best_loss:
            best_loss = val_loss
            best_ker = ker
            best_lr = lr
            best_lam = lam
        print "rate: %f, lam: %f, train : %f, verify: %f" %(lr, lam, rmse(ker, train, trainAns), val_loss)
#  best_ker = adagrad(kernel, np.append(train, validate), np.append(trainAns, validateAns), lr, lam, 4000)
print best_ker
print "Calculating result"


test = pd.read_csv('test_X.csv', sep=',', names=['id', 'Type'] + range(9)).replace('NR', 0)
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
              [1])
          for id, x in test.groupby('id') ])
print test[:5]
result = zip(testId, np.matmul(test, best_ker))
result.sort(key=lambda x: int(x[0][3:]))

