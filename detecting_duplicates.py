from enum import Enum
import math
import random
import itertools
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm
from scipy.spatial.distance import pdist
import scipy.integrate as integrate
# TEST AND DEBUG
import pdb
import unittest

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

class TestHitOrMissModule(unittest.TestCase):
    def setUp(self):
        df1 = pd.DataFrame(np.random.randint(100,size=(3*64,1)),
                           columns=['mg ingeridos'])
        df2 = pd.DataFrame(pd.date_range('2018-01-01', periods=3*64, freq='10 Min'),
                           columns=['horas'])
        df3 = expand_grid({'edad': [15, 20, 25],
                           'peso': [75, 80, 85, 90],
                           'genero': ['Masculino', 'Femenino', 'Neutro', 'Fluido'],
                           'nombre': ['Mario', 'Lola', 'Antonio', 'Pedro']})
        
        self.test = pd.concat([df1, df2, df3], axis=1)
        (m, n) = self.test.shape
        self.blank_frequency = dict() 
        for i, col in zip(range(n),self.test):
            for j in range(i):
                self.test.loc[j,col] = np.nan
            self.blank_frequency[col] = i/m
                 

        df4 = pd.DataFrame(np.random.randint(100,size=(16,1)),
                           columns=['mg ingeridos'])
        df5 = pd.DataFrame(pd.date_range('2018-01-01', periods=16, freq='10 Min'),
                           columns=['horas'])
        df6 = expand_grid({'edad': [15, 20],
                           'peso': [75, 90],
                           'genero': ['Masculino', 'Femenino'],
                           'nombre': ['Mario', 'Lola']})
        df7 =pd.DataFrame(np.arange(16),
                          columns=['pairnumber'])

        train = pd.concat([df4, df5, df6, df7], axis=1)
        self.train = pd.concat([train, train], axis=0)

        self.model = hom_model(self.train, self.test, [Algorithm.HOM]*4 + [Algorithm.MIXT]*2)

    def tearDown(self):
        self.model = hom_model(self.train, self.test, [Algorithm.HOM]*4 + [Algorithm.MIXT]*2)

    def test_blank_frequencies(self):
        self.assertEqual(self.model.bs, self.blank_frequency,
                         'incorrect frequency of blank elements')

    # Se puede tener un valor None dentro de un dataframe
    def test_nan_analog_for_strings(self):
        
    def test_normal_fits(self):
        

    # def test_frequency_every_element_appears(self):


    

class Algorithm(Enum):
    HOM = 0
    MIXT = 1

def delta(d):
    if d == 0:
        return 1
    else:
        return 0

class hom_model:

    def __init__(self, train, test, col_alg):
        # TODO: Anadir un vector de epsilon para aquellas varibles que puedan tener una desviacion.
        # TODO: Como se construyen train + test ?
        self.test = test
        # TODO: Revisar en el paper como se puede definir a
        self.col_alg = col_alg
        # TODO: Comprobar que siempre que se llame a bs se hace con col.
        self.bs = self.blank_freq()
        self.betas = self.rel_freq()
        self.fs = self.norm_fits()
        self.ds = self.differences()
        self.cs = self.dp_freq(train)
        self.a1s, self.a2s, self.sq_sigmas = self.calculate_frequencies(train)

    def blank_freq(self):
        (n,_) = self.test.shape
        return [1 - x/n for x in self.test.count()]

    def rel_freq(self):
        (n,_) = self.test.shape
        ret = dict()
        sel = [alg == Algorithm.HOM for alg in self.col_alg]        
        for col in self.test.loc[:,sel]:
            d = defaultdict(int)
            for x in self.test.loc[:,col]:
                d[x] += 1
            for k in d:
                d[k] /= n
            ret[col] = d
        return ret

    def norm_fits(self):
        ret = dict()
        sel = [alg == Algorithm.MIXT for alg in self.col_alg]
        for col in self.test.loc[:,sel]:
            xs = self.test.loc[:,col].to_numpy()
            ret[col] = norm.fit(xs[np.logical_not(np.isnan(xs))])
        return ret

    def differences(self):
        ret = dict()
        (m,_) = self.test.shape
        sel = [alg == Algorithm.MIXT for alg in self.col_alg]
        for col in self.test.loc[:,sel]:
            xs = test.loc[:,col].to_numpy()
            ret[col] = pdist(xs.reshape(m,1))
        return ret

    def dp_freq(self, train):
        (n, _) = train.shape
        ret = dict()
        grouped = train.groupby('pairnumber')
        sel = [alg == Algorithm.HOM for alg in self.col_alg]        
        for col in self.test.loc[:,sel]:
            ret[col] = 0
            for pair, group in grouped:
                elems = group.loc[:,col].to_numpy()
                elems = set(elems)
                if len(elems)==1:
                    ret[col] += 1
            s = 0
            for key in self.betas[col]:
                s += (self.betas[col][key])**2
            ret[col] = ret[col]/(n*(1-s))
        return ret
    
    def calc_as(self):
        ret = dict()
        sel = [alg == Algorithm.HOM for alg in self.col_alg]        
        for col in self.data.loc[:,sel]:
            b = self.bs[col]
            c = self.cs[col]
            # SOLVE X^2 + 2(b-1)X + c = 0 with 0 <= X <= 1
            a = self.get_correct_root(b, c)
            ret[col] = a
        return ret

    def get_correct_root(b,c):
        root1 = (-2*(b-1) + math.sqrt((-2*(b-1))**2 - 4*c))/2
        root2 = (-2*(b-1) - math.sqrt((-2*(b-1))**2 - 4*c))/2
        if 0 <= root1 <=1:
            return root1
        elif 0 <= root2 <=1:
            return root2
        else:
            raise Exception("Invalid probability")
        

    def calculate_frequencies(self, train):
        a1s, a2s, sq_sigmas = dict(), dict(), dict()
        sel = [alg == Algorithm.MIXT for alg in self.col_alg]
        cols = self.test.loc[:,sel].columns
        for col in train.loc[:,cols]:
            a1, a2, sq_sigma = self.hom_mix_fitting(self.ds[col])
            a1s[col] = a1
            a2s[col] = a2
            sq_sigmas[col] = sq_sigma
        return a1s, a2s, sq_sigmas

    def hom_mix_fitting(self, ds, b):
        eps = 0.001
        a1, a2 = random.random(), random.random()
        sq_sigma = np.var(ds)
        aux1, aux2, aux3 = 0, 0, 0
        while np.linalg.norm([a1-aux1, a2-aux2, sq_sigma]) > eps:
            a1, a2, sq_sigma = self.hom_mix_fitting_loop(ds, a1, a2, b, sq_sigma)
        return a1, a2, sq_sigma

    def gammas(self, col, di, alfa1, alfa2, alfa3, alfa4, sq_sigma):
        delta = delta(di)
        loc, scale = self.fs[col]
        f = norm.pdf(di, loc, scale)
        norm1 = norm.pdf(di, 0, sq_sigma)
        norm2 = norm.pdf(di, 0, 2*sq_sigma)
        gamma1 = alfa1*delta
        gamma2 = alfa2*f
        gamma3 = alfa3*norm1
        gamma4 = alfa4*norm2
        den = gamma1 + gamma2 + gamma3 + gamma4
        return gamma1/den, gamma2/den, gamma3/den, gamma4/den

    def alfa1(a1, a2, b):
        return (1-a1-a2-b)**2

    def alfa2(a2, b):
        return a2*(2-2*b-a2)

    def alfa3(a1, a2, b):
        return 2*a1*(1-a1-a2-b)

    def alfa4(a1):
        return a1**2

    def hom_mix_fitting_loop(self, ds, a1, a2, b, sq_sigma):
        alfa1 = alfa1(a1, a2, b)
        alfa2 = alfa2(a2, b)
        alfa3 = alfa3(a1, a2, b)
        alfa4 = alfa4(a1)

        gamma3s = np.array([])
        gamma4s = np.array([])
        
        for d in diffs:
            _, _, gamma3, gamma4 = self.gammas(d, alfa1, alfa2, alfa3, alfa4, sq_sigma)
            gamma3s.append([gamma3])
            gamma4s.append([gamma4])

        num = np.sum((gamma3s + gamma4s*0.5)*diffs**2)
        den = np.sum(gamma3s + gamma4s)
        sq_sigma = num/den

        a1, a2 = self.update_as(a1, a2, b, d, sq_sigma)

        return a1, a2, sq_sigma

    def update_as(self, a1, a2, b, d, sq_sigma):
        x0 = [a1, a2]
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] < 1-b})
        bnds ((0, 1), (0,1))
        fun = lambda x: - self.expected_likelihood(x, b, d, sq_sigma)
        res = minimize(fun, x0)
        return res[0], res[1]

    def expected_likelihood(self, x, b, d, sq_sigma):
        a1 = x[0]
        a2 = x[1]
        return np.prod([self.prob1(a1,a2,b,d,sq_sigma) for d in ds])

    def prob1(self, a1, a2, b, d, sq_sigma, fd):
        if np.isnan(d):
            return 1 - (1 - b)**2
        else:
            k1 = alfa1(a1,a2,b)
            di = delta(d)

            k2 = alfa2(a2,b)
            n1 = norm.pdf(d, 0, sq_sigma)

            k3 = alfa3(a1,a2,b)
            n2 = norm.pdf(d, 0, 2*sq_sigma)

            k4 = alfa4(a1)
            loc, scale = fd
            f = norm.pdf(d, loc, scale)

            return k1*di + k2*f + k3*n1 + k4*n2

    def prob2(self, b, d, fd):
        if np.isnan(d):
            return 1 - (1-b)**2
        else:
            loc, scale = fd
            f = norm.pdf(d, loc, scale)
            return (1-b)**2*f

    def wkj(self, j, k):
        ret = 0
        (_,n) = self.test.shape
        for i, col in zip(range(n), test):
            if self.col_alg[i] == Algorithm.HOM:
                ret += self.wjk_hom(j, k, col)
            elif self.col_alg[i] == Algorithm.MIXT:
                ret += self.wjk_mixt(j, k, col)
        return ret

    def wjk_hom(self, j, k, col):
        x = self.test.loc[j, col]
        y = self.test.loc[k, col]
        b = self.bs[col]
        c = self.cs[col]
        beta = self.betas[col][j]
        if np.isnan(x) or np.isnan(y):
            return 0
        elif np.logical_and(np.equal(x, y)):
            return np.log2(1 - c*(1 - beta)*(1 - b)**(-2)) - np.log2(beta)
        else:
            return np.log2(c) - 2*np.log2(1-b)

    def wjk_mixt(self, j, k, col):
        a1 = self.a1s[col]
        a2 = self.a2s[col]
        b = self.bs[col]
        d = self.test.loc[j, col] - self.test.loc[k, col]
        epsilon = self.epsilons[col]
        s1 = self.sq_sigmas[col]
        fd = self.fs[col]
        if np.isnan(d):
            return 0
        else:
            fun1 = lambda x : self.prob1(a1, a2, b, d, s1, fd)
            fun1 = lambda x : self.prob2(b, d, fd)
            i1 = integrate.quad(fun1, d-epsilon, d+epsilon)
            i2 = integrate.quad(fun2, d-epsilon, d+epsilon)
            return np.log2(i1/i2)

            
if __name__=="__main__":
    unittest.main()
