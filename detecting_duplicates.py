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
    """Builds a pandas dataframe with some values

    Args:
        data_dict: A dictionary with keys and values

    Returns:
        A pandas dataframe where the columns are the keys 
        of the dictionary and each row is any of the possible
        combinations of the values for each key.
    """
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

class TestHitOrMissModule(unittest.TestCase):
    def setUp(self):
        df1 = pd.DataFrame(np.random.randint(100,size=(3*64,1)),
                           columns=['mg ingeridos'])
        # TODO: Como automatizar que todos los datos de DEV sean numericos???
        df2 = pd.DataFrame(pd.to_numeric(pd.date_range('2018-01-01', periods=3*64, freq='10 Min')),
                           columns=['horas'])
        df3 = expand_grid({'edad': [15, 20, 25],
                           'peso': [75, 80, 85, 90],
                           'genero': ['Masculino', 'Femenino', 'Neutro', 'Fluido'],
                           'nombre': ['Mario', 'Lola', 'Antonio', 'Pedro']})
        epsilon = {'mg ingeridos': 30,
                   'horas': 100,
                   'edad': 1,
                   'peso': 5}
        self.test = pd.concat([df1, df2, df3], axis=1)
        (m, n) = self.test.shape
        self.blank_frequency = dict() 
        for i, col in zip(range(n),self.test):
            for j in range(min(i,m)):
                self.test.loc[j,col] = np.nan
            self.blank_frequency[col] = i/m
                 

        df4 = pd.DataFrame(np.random.randint(100,size=(16,1)),
                          columns=['mg ingeridos'])
        df5 = pd.DataFrame(pd.to_numeric(pd.date_range('2018-01-01', periods=16, freq='10 Min')),
                           columns=['horas'])
        df6 = expand_grid({'edad': [15, 20],
                           'peso': [75, 90],
                           'genero': ['Masculino', 'Femenino'],
                           'nombre': ['Mario', 'Lola']})
        df7 =pd.DataFrame(np.arange(16),
                          columns=['pairnumber'])

        train = pd.concat([df4, df5, df6, df7], axis=1)
        self.train = pd.concat([train, train], axis=0)
        
        alg_selection = [Algorithm.DEV]*4 + [Algorithm.HOM]*2
        self.model = hom_model(self.train, self.test, alg_selection, epsilon)

    def test_blank_frequencies(self):
        self.assertEqual(self.model.bs, self.blank_frequency,
                         'incorrect frequency of blank elements')

    # Se puede tener un valor None dentro de un dataframe
    # def test_nan_analog_for_strings(self):
        
    def test_normal_fits(self):
        d = dict()
        sel = [alg == Algorithm.DEV for alg in self.model.col_alg]
        for col in self.test.loc[:,sel]:
            B = self.test.loc[:,col].to_numpy()
            withoutnans = B[np.logical_not(np.isnan(B))]
            d[col] = norm.fit(withoutnans)
        self.assertEqual(d, self.model.fs)
            

    # def test_frequency_every_element_appears(self):


    

class Algorithm(Enum):
    HOM = 0
    DEV = 1

class hom_model:
    """Hit or miss model to estimate how similar two rows in a database are

    Atributes:
       test: dataframe that contains the database
       col_alg: array of Algorithm describing the algorithm used for each column
       bs: dict where each column is associated to its frequency of blanks
       betas: dict where each column is associated to the frequency of each value
       fs: dict containing mean and var. of a normal fit to each column
       ds: dict containing an array of differences of values inside each column
       a1s: EM estimates of parameter a1 for each DEV column 
       a2s: EM estimates of parameter a2 for each DEV column       
       sq_sigmas: EM estimates of parameter sq_sigma for each DEV column       
    """
    def __init__(self, train, test, col_alg, epsilon):
        """Initialize hit or miss probabilistic model
        
        Args:
            train: dataframe used to train the model
            test: dataframe with the data
            col_alg: array specifying the algorithm used for each column
            epsilon: dictionary that stores possible deviations that the data inside a column can have
            
        Returns:
            A hom model for estimating how similar two rows of the data are
        """
        self.test = test
        self.col_alg = col_alg
        self.epsilon = epsilon
        self.bs = self.blank_freq()
        self.betas = self.rel_freq()
        self.fs = self.norm_fits()
        self.ds = self.differences(train)
        self.cs = self.dp_freq(train)
        self.a1s, self.a2s, self.sq_sigmas = self.calculate_frequencies(train)

    def blank_freq(self):
        """Returns the blank frequencies for each column"""
        (n,_) = self.test.shape
        ret = dict()
        for col, x in zip(self.test.columns,self.test.count()):
            ret[col] = 1 - x/n
        return ret

    def rel_freq(self):
        """Returns the frequency for each value inside each column"""
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
        """Fits a normal distribution for each column and returns sufficient statisticals"""
        ret = dict()
        sel = [alg == Algorithm.DEV for alg in self.col_alg]
        for col in self.test.loc[:,sel]:
            xs = self.test.loc[:,col].to_numpy()
            withoutnans = xs[np.logical_not(np.isnan(xs))]
            ret[col] = norm.fit(withoutnans)
        return ret

    def differences(self, train):
        """Calculates pairwise element difference for each numerical column

        Args:
            train: dataframe containing the train data

        Returns:
            A dict containing for each column all the possible differences between the data.
        """
        df = train.loc[:,self.test.columns]
        ret = dict()
        (m,_) = df.shape
        sel = [alg == Algorithm.DEV for alg in self.col_alg] 
        for col in df.loc[:,sel]:
            xs = df.loc[:,col].to_numpy()
            ret[col] = pdist(xs.reshape(m,1))
        return ret

    def dp_freq(self, train):
        """Calculates the frequency of discordant pairs for each column in the identified duplicates
        
        Args:
            train: dataframe containing the train data

        Returns:
            A dict containing for each column frequencies of discordant pairs in identified duplicates.
        """
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
        """Calculate probability of miss for each column"""
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
        """Gives a solution for X^2 + 2(b-1)X + c = 0 with 0 <= X <= 1"""
        root1 = (-2*(b-1) + math.sqrt((-2*(b-1))**2 - 4*c))/2
        root2 = (-2*(b-1) - math.sqrt((-2*(b-1))**2 - 4*c))/2
        if 0 <= root1 <=1:
            return root1
        elif 0 <= root2 <=1:
            return root2
        else:
            raise Exception("Invalid probability")
        

    def calculate_frequencies(self, train):
        """Calculate estimates for HOM DEV algorithm"""
        a1s, a2s, sq_sigmas = dict(), dict(), dict()
        sel = [alg == Algorithm.DEV for alg in self.col_alg]
        cols = self.test.loc[:,sel].columns
        for col in train.loc[:,cols]:
            a1, a2, sq_sigma = self.hom_mix_fitting(col)
            a1s[col] = a1
            a2s[col] = a2
            sq_sigmas[col] = sq_sigma
        return a1s, a2s, sq_sigmas
    
    def hom_mix_fitting(self, col):
        ds = self.ds[col]
        sq_sigma = np.var(ds)
        a1, a2 = random.random(), random.random()
        aux1, aux2, aux3 = 0, 0, 0
        eps = 0.001
        while np.linalg.norm([a1-aux1, a2-aux2, sq_sigma-aux3]) > eps:
            aux1, aux2, aux3 = a1, a2, sq_sigma
            a1, a2, sq_sigma = self.hom_mix_fitting_loop(col, a1, a2, sq_sigma)
            print(np.linalg.norm([a1-aux1, a2-aux2, sq_sigma-aux3]))
        return a1, a2, sq_sigma

    def hom_mix_fitting_loop(self, col, a1, a2, sq_sigma):
        b = self.bs[col]
        ds = self.ds[col]

        gamma1s = np.array([])
        gamma2s = np.array([])
        gamma3s = np.array([])
        gamma4s = np.array([])

        for d in ds:
            gamma1, gamma2, gamma3, gamma4 = self.gammas(col, d, a1, a2, b, sq_sigma)
            gamma1s = np.append(gamma1s,[gamma1])
            gamma2s = np.append(gamma2s,[gamma2])
            gamma3s = np.append(gamma3s,[gamma3])
            gamma4s = np.append(gamma4s,[gamma4])
            
        num = np.sum((gamma3s + gamma4s*0.5)*ds**2)
        den = np.sum(gamma3s + gamma4s)
        sq_sigma = num/den

        a1, a2 = self.update_as(col, a1, a2, b, ds, sq_sigma, gamma1s, gamma2s, gamma3s, gamma4s)

        return a1, a2, sq_sigma

    def alfa1(a1, a2, b):
        return (1-a1-a2-b)**2

    def alfa2(a2, b):
        return a2*(2-2*b-a2)

    def alfa3(a1, a2, b):
        return 2*a1*(1-a1-a2-b)

    def alfa4(a1):
        return a1**2

    def delta(d):
        if d == 0:
            return 1
        else:
            return 0
    
    def gammas(self, col, di, a1, a2, b, sq_sigma):
        alfa1 = hom_model.alfa1(a1, a2, b)
        alfa2 = hom_model.alfa2(a2, b)
        alfa3 = hom_model.alfa3(a1, a2, b)
        alfa4 = hom_model.alfa4(a1)
        delta = hom_model.delta(di)
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

    def update_as(self, col, a1, a2, b, ds, sq_sigma, gamma1s, gamma2s, gamma3s, gamma4s):
        x0 = [a1, a2]
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] < 1-b})
        bnds = ((0, 1), (0,1))
        fun = lambda x: - self.expected_likelihood(col, x, b, ds, sq_sigma, gamma1s, gamma2s, gamma3s, gamma4s)
        res = minimize(fun, x0)
        return res.x[0], res.x[1]

    def expected_likelihood(self, col, x, b, ds, sq_sigma, gamma1s, gamma2s, gamma3s, gamma4s):
        a1 = x[0]
        a2 = x[1]
        comp1 = np.array([])
        comp2 = np.array([])
        comp3 = np.array([])
        comp4 = np.array([])
        loc, scale = self.fs[col]
        for d in ds:
            k1 = hom_model.alfa1(a1,a2,b)
            di = hom_model.delta(d)
            l1 = np.log2(k1*di)
            comp1 = np.append(comp1, [l1])

            k2 = hom_model.alfa2(a2,b)
            n1 = norm.pdf(d, 0, sq_sigma)
            l2 = np.log2(k2*n1)
            comp2 = np.append(comp2, [l2])

            k3 = hom_model.alfa3(a1,a2,b)
            n2 = norm.pdf(d, 0, 2*sq_sigma)
            l3 = np.log2(k2*n1)
            comp3 = np.append(comp3, [l3])

            k4 = hom_model.alfa4(a1)
            f = norm.pdf(d, loc, scale)
            l4 = np.log2(k4*f)
            comp4 = np.append(comp4, [l4])

        ret = gamma1s*comp1
        ret += gamma2s*comp2
        ret += gamma3s*comp3
        ret += gamma4s*comp4
        return np.sum(ret)

    def prob1(self, col, a1, a2, b, d, sq_sigma):
        loc, scale = self.fs[col]
        if np.isnan(d):
            return 1 - (1 - b)**2
        else:
            k1 = hom_model.alfa1(a1,a2,b)
            di = hom_model.delta(d)

            k2 = hom_model.alfa2(a2,b)
            n1 = norm.pdf(d, 0, sq_sigma)

            k3 = hom_model.alfa3(a1,a2,b)
            n2 = norm.pdf(d, 0, 2*sq_sigma)

            k4 = hom_model.alfa4(a1)
            f = norm.pdf(d, loc, scale)

            return k1*di + k2*f + k3*n1 + k4*n2

    def prob2(self, col, b, d):
        loc, scale = self.fs[col]
        if np.isnan(d):
            return 1 - (1-b)**2
        else:
            f = norm.pdf(d, loc, scale)
            return (1-b)**2*f

    def wkj(self, j, k):
        ret = 0
        (_,n) = self.test.shape
        for i, col in zip(range(n), test):
            if self.col_alg[i] == Algorithm.HOM:
                ret += self.wjk_hom(j, k, col)
            elif self.col_alg[i] == Algorithm.DEV:
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
            fun1 = lambda x : self.prob1(col, a1, a2, b, d, s1)
            fun1 = lambda x : self.prob2(col, b, d)
            i1 = integrate.quad(fun1, d-epsilon, d+epsilon)
            i2 = integrate.quad(fun2, d-epsilon, d+epsilon)
            return np.log2(i1/i2)

            
if __name__=="__main__":
    unittest.main()
