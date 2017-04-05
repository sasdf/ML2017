import numpy as np, pandas as pd
import functools as fn
import math
import sys
import pickle
# Filter feature
def transform(df):
    for n in ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']:
        pass
        df['cont_'+n] = df[n]
        df = df.drop(n, axis=1)
        n = 'cont_' + n
        #  df[n+'**0.125'] = (df[n]-df[n].min())**0.125
        #  df[n+'**0.25'] = (df[n]-df[n].min())**0.25
        #  df[n+'**0.5'] = (df[n]-df[n].min())**0.5
        #  df[n+'log'] = (df[n]-df[n].min()+1).map(math.log)
        #  for p in range(2,3):
            #  df[n+'**'+str(p)] = df[n]**p
    #  df['age**2'] = df['capital_gain']**2
    return df
def extract(df):
    #  df['PM2.5s2'] = x**0.15
    df = transform(df)
    org = transform(pd.read_csv(sys.argv[1], sep=',', header='infer'))
    for n in [ x for x in df.columns if x.startswith('cont_')]:
        x = org[n]
        df[n] = (df[n] - x.mean()) / x.std()
        #  df[n] = (df[n] - x.mean()) / x.std()
    #  df = df.drop('age', axis=1)
    df = df.sort_index(axis=1)
    return df

def Gau(x, cov):
    c = np.matmul(cov.T, cov).sum().sum()**0.5
    a = np.matmul(x, np.linalg.inv(cov))
    b = np.multiply(a, x).sum(axis=1)
    return np.exp(-b/2)/(c**0.5)/((2*np.pi)**(len(cov)/2))

seed = 6
np.random.seed(seed)

# Read csv
df = pd.read_csv(sys.argv[1], sep=',', header='infer')
dfy = pd.read_csv(sys.argv[2], sep=',', header=None, names=['Ans'])

# Reshape

df = extract(df)
df = pd.concat([df, dfy], axis=1)
df0 = df[df['Ans'] == 0].drop('Ans', axis=1)
df1 = df[df['Ans'] == 1].drop('Ans', axis=1)
df0c = df0[[ x for x in df0.columns if x.startswith('cont_')]].values
df0d = df0[[ x for x in df0.columns if not x.startswith('cont_')]].values
df1c = df1[[ x for x in df1.columns if x.startswith('cont_')]].values
df1d = df1[[ x for x in df1.columns if not x.startswith('cont_')]].values
co0 = np.cov(df0c, rowvar = False)
co1 = np.cov(df1c, rowvar = False)
p0 = len(df0) * 1.0 / len(df)
p1 = 1.0 - p0
pd0 = df0d.sum(axis=0).astype(float) / len(df)
pd1 = df1d.sum(axis=0).astype(float) / len(df)
#  print co0
#  print co1
test = pd.read_csv(sys.argv[3], sep=',', header='infer')
test = extract(test).astype(float)

tc = test[[ x for x in test.columns if x.startswith('cont_')]].values
td = test[[ x for x in test.columns if not x.startswith('cont_')]].values

tpd0 = (td * (1 - pd0*2) + pd0)+1
tpd1 = (td * (1 - pd1*2) + pd1)+1
px0 = Gau(tc, co0) * (tpd0.prod(axis=1)+1)
px1 = Gau(tc, co1) * (tpd1.prod(axis=1)+1)
print 'a'
print px0[:10]
print px1[:10]

p1x = px1*p1/(px0*p0+px1*p1)
p0x = px0*p0/(px0*p0+px1*p1)
print p1x[:10]
print p0x[:10]
e = np.round(p1x)
print e[:10]

result = zip(test.index, e)
with open(sys.argv[4], 'w') as f:
    f.write('id,label\n' + ''.join(['%d,%d\n'%(int(id)+1, int(val)) for id, val in result]))
