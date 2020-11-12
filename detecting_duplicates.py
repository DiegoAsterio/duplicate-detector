# TODO: De verdad hace falta un enumerable para un valor Booleano?
from enum import Enum, auto

import itertools

import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm
from scipy.spatial.distance import pdist
import scipy.integrate as integrate

from matplotlib.pyplot import plot

# TEST AND DEBUG
import unittest
import random
import pdb
import warnings

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
    
    # def setUp(self):
        # df1 = pd.DataFrame(np.random.randint(100, size=(3 * 64, 1)),
        #                    columns=['mg ingeridos'])
        # # TODO: Como automatizar que todos los datos de DEV sean numericos???
        # df2 = pd.DataFrame(pd.to_numeric(
        #     pd.date_range('2018-01-01', periods=3 * 64, freq='10 Min')),
        #                    columns=['horas'])
        # df3 = expand_grid({
        #     'edad': [15, 20, 25],
        #     'peso': [75, 80, 85, 90],
        #     'genero': ['Masculino', 'Femenino', 'Neutro', 'Fluido'],
        #     'nombre': ['Mario', 'Lola', 'Antonio', 'Pedro']
        # })
        # epsilon = {'mg ingeridos': 30, 'horas': 100, 'edad': 1, 'peso': 5}
        # self.data = pd.concat([df1, df2, df3], axis=1)
        # self.data = pd.concat([self.data, self.data],
        #                       axis=0,
        #                       ignore_index=True)
        # (m, n) = self.data.shape
        # self.blank_frequency = dict()
        # for i, col in enumerate(self.data):
        #     for j in range(min(i, m)):
        #         self.data.loc[j, col] = np.nan
        #     self.blank_frequency[col] = i / m

        # self.train = pd.DataFrame(np.array(
        #     [[str(x), str(int((x + m / 2) % m))] for x in range(m)]),
        #                           columns=["Issue_Id", "Duplicate"])

        # alg_selection = [Algorithm.DEV] * 4 + [Algorithm.HOM] * 2
        # self.model = hom_model(self.data, self.train, alg_selection, epsilon)

    def test_blank_frequencies(self):
        df = pd.DataFrame({'A': [np.nan, 1.0, 1.0, 1.0],
                           'B': [np.nan, np.nan, 1.0, 1.0],
                           'C': [np.nan, np.nan, np.nan, 1.0],
                           'D': [np.nan, np.nan, np.nan, np.nan]})
        model = hom_model(df, None, [Algorithm.HOM]*4, None)
        bf = {'A': 0.25, 'B': 0.5, 'C': 0.75, 'D': 1.0}
        self.assertEqual(model.bs, bf,
                         'incorrect frequency of blank elements')

    def test_relative_frequencies(self):
        df = pd.DataFrame({'A': [1.0, 1.0, 1.0, 1.0],
                           'B': [1.0, 1.0, 2.0, 2.0],
                           'C': [1.0, 2.0, 2.0, 3.0],
                           'D': [1.0, 2.0, 3.0, 4.0]})
        model = hom_model(df, None, [Algorithm.HOM]*4, None)
        rf = {'A': {1.0: 1.0},
              'B': {1.0: 0.5,
                    2.0: 0.5},
              'C': {1.0: 0.25,
                    2.0: 0.5,
                    3.0: 0.25},
              'D': {1.0: 0.25,
                    2.0: 0.25,
                    3.0: 0.25,
                    4.0: 0.25}}
        self.assertEqual(model.betas, rf,
                         'incorrect relative frequency of elements')
        
    # TODO: Ahora las diferencias se calculan solo sobre el test
    def test_calculate_differences(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0],
                           'B': [5.0, 5.0, 6.0, 7.0]})
        train = pd.DataFrame({'Issue_id': ["0", "1", "2", "3"],
                              'Duplicate': ["1", "0", "3", "2"]})
        model = hom_model(df, train, [Algorithm.DEV]*2, None)
        model_ds = dict()
        for k in model.ds:
            model_ds[k] = list(model.ds[k])
        ds = {'A': [-1.0,-2.0,-3.0,-1.0,-2.0,-1.0],
              'B': [0.0,-1.0,-2.0,-1.0,-2.0,-1.0]}
        
        self.assertEqual(model_ds, ds,
                         'incorrect differences')

        
    def test_discordant_pairs_frequency(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0],
                           'B': [5.0, 5.0, 6.0, 7.0]})
        train = pd.DataFrame({'Issue_id': ["0", "1", "2", "3"],
                              'Duplicate': ["1", "0", "3", "2"]})
        model = hom_model(df, train, [Algorithm.HOM]*2, None)
        dp = {'A': 2/3, 'B': 2/5}
        self.assertEqual(model.cs, dp,
                         'incorrect discordant pairs freq.')
        
        
        
    # Se puede tener un valor None dentro de un dataframe
    # def test_nan_analog_for_strings(self):

    # def test_normal_fits(self ):
    #     d = dict()
    #     sel = [alg == Algorithm.DEV for alg in self.model.algs]
    #     for col in self.data.loc[:,sel]:
    #         B = self.data.loc[:,col].to_numpy()
    #         withouans = B[np.logical_not(np.isnan(B))]
    #         d[col] = norm.fit(withoutnans)
    #     self.assertEqual(d, self.model.fs)

    # def test_frequency_every_element_appears(self):


class Algorithm(Enum):
    HOM = auto()
    DEV = auto()


class hom_model:
    """Hit or miss model to estimate how similar two rows in a database are

    Atributes:
       data: dataframe that contains the database
       algs: array of Algorithm describing the algorithm used for each column
       bs: dict where each column is associated to its frequency of blanks
       betas: dict of dicts where each column is associated to a dict containing
       the frequency of each value
       fs: dict containing mean and var. of a normal fit to each column
       ds: dict containing an array of differences of values inside each column
       a1s: EM estimates of parameter a1 for each DEV column 
       a2s: EM estimates of parameter a2 for each DEV column       
       sq_sigmas: EM estimates of parameter sq_sigma for each DEV column       
    """
    def __init__(self, data, train, algs, eps):
        """Initialize hit or miss probabilistic model
        
        Args:
            data: dataframe with the data
            train: dataframe used to train the model
            algs: array specifying the algorithm used for each column
            eps: dictionary that stores possible deviations that the data inside a column can have
            
        Returns:
            A hom model for estimating how similar two rows of the data are
        """
        self.data = data
        self.algs = algs
        self.eps = eps
        self.bs = self.blank_freq()
        self.betas = self.rel_freq()
        self.fs = self.norm_fits(train)
        self.ds = self.differences(train)
        self.cs = self.dp_freq(train)
        self.a1s, self.a2s, self.sq_sigmas1, self.sq_sigmas2 = self.calculate_frequencies()
        self.eps = eps
        # UN GASTO ENORME DE MEMORIA
        self.isna = self.data.isna()
        # self.t = self.calc_thres(train)

    def blank_freq(self):
        """Returns the blank frequencies for each column"""
        (n, _) = self.data.shape
        calc_bf = lambda x: 1 - x / n
        return self.data.count().apply(calc_bf).to_dict()

    def cols(self, alg):
        """Returns the columns where a certain algorithm is executed"""
        it = zip(self.data.columns, self.algs)
        return (col for col, a in it if a == alg)

    def get_fs(self, col):
        df = self.data
        (n, _) = df.shape  # nxm -> n
        norm = lambda x: x / n
        # TODO: Numba Accelerated Routine
        return df.groupby(col).size().apply(norm).to_dict()
    
    def rel_freq(self):
        """Returns the frequency for each value inside each column"""
        cols = self.cols(Algorithm.HOM)
        return {col: self.get_fs(col) for col in cols}

    def get_ds(self, index_sel, col):
        xs = self.data.loc[index_sel, col].dropna().to_numpy()
        return pdist(xs.reshape(len(xs), 1), lambda x, y: x-y)
    
    def norm_fits(self, train):
        """Fits a normal distribution to the differences of the elemenst in a numerical column"""
        try:
            train.shape
        except AttributeError as N:
            err_msg ="Incorrect use of function differences:\n"
            print(err_msg, self.differences.__doc__)
            return None
        
        train_data = train.dropna().iloc[:, 0].apply(int)
        cols = self.cols(Algorithm.DEV)
        return {col: norm.fit(self.get_ds(train_data, col)) for col in cols}

    def differences(self, train):
        """Calculates pairwise element difference for each numerical column

        Args:
            train: dataframe containing the train data

        Returns:
            A dict containing for each column all the possible differences between the data.
        """
        try:
            train.shape
        except AttributeError as N:
            err_msg ="Incorrect use of function differences:\n"
            print(err_msg, self.differences.__doc__)
            return None

        ret = dict()
        vals = train.dropna().iloc[:, 0].apply(int)
        df = self.data.loc[vals]
        for col in self.cols(Algorithm.DEV):
            xs = df.loc[:, col].dropna().to_numpy()
            xs = xs.reshape(len(df), 1)
            ret[col] = pdist(xs, lambda x, y: x - y)
        return ret

    def dp_freq(self, train):
        """Calculates the frequency of discordant pairs for each column in the identified duplicates
        
        Args:
            train: dataframe containing the train data

        Returns:
            A dict containing for each column frequencies of discordant pairs in identified duplicates.
        """
        try:
            train.shape
        except AttributeError as N:
            err_msg ="Incorrect use of function dp_freq:\n"
            print(err_msg, self.dp_freq.__doc__)
            return None

        groups = train.dropna()  # Drop rows with invalid values
        (n, _) = groups.shape

        cols = self.cols(Algorithm.HOM)
        ret = {col: 0 for col in cols}
        for _, row in groups.iterrows():
            i = int(row[0])  # StringToInt to get i
            js = [int(x)
                  for x in row[1].split(';')]  # [StringToInt] to get [j]
            for j in js:
                for col in cols:
                    if self.data.loc[i, col] != self.data.loc[j, col]:
                        # Notice that every discordant pair DP is
                        # added twice. First when chosen by (i,j)
                        # and second when chosen by (j,i)
                        ret[col] += 1  # Adds a new discordant pair
        for col in cols:
            s = 0
            for key in self.betas[col]:
                s += (self.betas[col][key])**2
            # The two is due to the fact that DPs are added twice
            ret[col] /= 2 * n * (1 - s)
        return ret

    # def calc_as(self):
    #     """Calculate probability of miss for each column"""
    #     ret = dict()
    #     for col in self.cols(Algorithm.HOM):
    #         b = self.bs[col]
    #         c = self.cs[col]
    #         # SOLVE X^2 + 2(b-1)X + c = 0 with 0 <= X <= 1
    #         a = self.get_correct_root(b, c)
    #         ret[col] = a
    #     return ret

    # def get_correct_root(b, c):
    #     """Gives a solution for X^2 + 2(b-1)X + c = 0 with 0 <= X <= 1"""
    #     root1 = (-2 * (b - 1) + np.sqrt((-2 * (b - 1))**2 - 4 * c)) / 2
    #     root2 = (-2 * (b - 1) - np.sqrt((-2 * (b - 1))**2 - 4 * c)) / 2
    #     if 0 <= root1 <= 1:
    #         return root1
    #     elif 0 <= root2 <= 1:
    #         return root2
    #     else:
    #         raise Exception("Invalid probability")

    def calculate_frequencies(self):
        """Calculate estimates for HOM DEV algorithm"""
        a1s, a2s, sq_sigma1s, sq_sigma2s = dict(), dict(), dict(), dict()
        for col in self.cols(Algorithm.DEV):
            a1, a2, sq_sigma1, sq_sigma2 = self.hom_mix_fitting(col)
            a1s[col] = a1
            a2s[col] = a2
            sq_sigma1s[col] = sq_sigma1
            sq_sigma2s[col] = sq_sigma2
        return a1s, a2s, sq_sigma1s, sq_sigma2s

    def hom_mix_fitting(self, col):
        """Computes the a1, a2 and variance for a column of the dataset
        
        Args:
            col: The column where the hit or miss deviation algorithm 
            calculates constants
            
        Returns:
            A triple with the EM algorithm output for a1, a2 and variance
        """
        ds = self.ds[col]
        sq_sigma2 = np.var(ds)
        sq_sigma1 = 0.001*sq_sigma2
        b = self.bs[col]
        a1 = (1-b)*random.random()
        a2 = (1 - b - a1)*random.random()
        aux1, aux2, aux3, aux4 = 0, 0, 0, 0
        eps = 0.001
        v = [a1 - aux1, a2 - aux2, sq_sigma1 - aux3, sq_sigma2 - aux4]
        d0 = np.linalg.norm(v)
        d1 = d0
        while d1 > eps:
            print(np.linalg.norm(v))
            aux1, aux2, aux3, aux4 = a1, a2, sq_sigma1, sq_sigma2
            a1, a2, sq_sigma1, sq_sigma2 = self.hom_mix_fitting_loop(col, a1, a2, sq_sigma1, sq_sigma2)
            v = [a1 - aux1, a2 - aux2, sq_sigma1 - aux3, sq_sigma2 - aux4]
            d0, d1 = d1, np.linalg.norm(v)
            if d0 == d1:
                coord = lambda x: 100 * x - 50
                a1 += coord(np.random.random())
                a2 += coord(np.random.random())
                sq_sigma1 += coord(np.random.random())
                sq_sigma2 += coord(np.random.random())
        return a1, a2, sq_sigma1, sq_sigma2

    def hom_mix_fitting_loop(self, col, a1, a2, sq_sigma1, sq_sigma2):
        b = self.bs[col]
        ds = self.ds[col]

        gamma1s = np.array([])
        gamma2s = np.array([])
        gamma3s = np.array([])
        gamma4s = np.array([])

        for d in ds:
            gamma1, gamma2, gamma3, gamma4 = self.gammas(
                col, d, a1, a2, b, sq_sigma1, sq_sigma2)
            gamma1s = np.append(gamma1s, [gamma1])
            gamma2s = np.append(gamma2s, [gamma2])
            gamma3s = np.append(gamma3s, [gamma3])
            gamma4s = np.append(gamma4s, [gamma4])

        num1 = np.sum(gamma1s * ds**2)
        den1 = np.sum(gamma1s)
        sq_sigma1 = num1 / den1

        num2 = np.sum((gamma3s + gamma4s * 0.5) * ds**2)
        den2 = np.sum(gamma3s + gamma4s)
        sq_sigma2 = num2 / den2

        a1, a2 = self.update_as(col, a1, a2, b, ds, sq_sigma1, sq_sigma2, gamma1s, gamma2s,
                                gamma3s, gamma4s)

        return a1, a2, sq_sigma1, sq_sigma2

    def alfa1(a1, a2, b):
        return (1 - a1 - a2 - b)**2

    def alfa2(a2, b):
        return a2 * (2 - 2 * b - a2)

    def alfa3(a1, a2, b):
        return 2 * a1 * (1 - a1 - a2 - b)

    def alfa4(a1):
        return a1**2

    def gammas(self, col, di, a1, a2, b, sq_sigma1, sq_sigma2):
        alfa1 = hom_model.alfa1(a1, a2, b)
        alfa2 = hom_model.alfa2(a2, b)
        alfa3 = hom_model.alfa3(a1, a2, b)
        alfa4 = hom_model.alfa4(a1)

        norm1 = norm.pdf(di, 0, sq_sigma1)

        loc, scale = self.fs[col]
        f = norm.pdf(di, loc, scale)

        norm2 = norm.pdf(di, 0, sq_sigma2)

        norm3 = norm.pdf(di, 0, 2 * sq_sigma2)

        gamma1 = alfa1 * norm1
        gamma2 = alfa2 * f
        gamma3 = alfa3 * norm2
        gamma4 = alfa4 * norm3

        den = gamma1 + gamma2 + gamma3 + gamma4

        return gamma1 / den, gamma2 / den, gamma3 / den, gamma4 / den

    def update_as(self, col, a1, a2, b, ds, sq_sigma1, sq_sigma2, gamma1s, gamma2s,
                  gamma3s, gamma4s):
        x0 = [a1, a2]
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] < 1 - b})
        bnds = ((0, 1), (0, 1))
        fun = lambda x: -self.expected_likelihood(
            col, x, b, ds, sq_sigma1, sq_sigma2, gamma1s, gamma2s, gamma3s, gamma4s)
        res = optimize.minimize(fun, x0, bounds=bnds, constraints=cons)
        print("The expected likelihood is: ", -fun(res.x))
        return res.x[0], res.x[1]

    def expected_likelihood(self, col, x, b, ds, sq_sigma1, sq_sigma2, gamma1s, gamma2s,
                            gamma3s, gamma4s):
        a1 = x[0]
        a2 = x[1]
        comp1 = np.array([])
        comp2 = np.array([])
        comp3 = np.array([])
        comp4 = np.array([])
        loc, scale = self.fs[col]
        pi1 = sum(gamma1s)/len(gamma1s)
        pi2 = sum(gamma2s)/len(gamma2s)
        pi3 = sum(gamma3s)/len(gamma3s)
        pi4 = sum(gamma4s)/len(gamma4s)

        k1 = hom_model.alfa1(a1, a2, b)
        k2 = hom_model.alfa2(a2, b)
        k3 = hom_model.alfa3(a1, a2, b)
        k4 = hom_model.alfa4(a1)
        for d in ds:
            di = norm.pdf(d, 0, sq_sigma1)
            l1 = np.log2(k1 * di)
            comp1 = np.append(comp1, [l1])

            n1 = norm.pdf(d, 0, sq_sigma2)
            l2 = np.log2(k2 * n1)
            comp2 = np.append(comp2, [l2])

            n2 = norm.pdf(d, 0, 2 * sq_sigma2)
            l3 = np.log2(k3 * n2)
            comp3 = np.append(comp3, [l3])

            f = norm.pdf(d, loc, scale)
            l4 = np.log2(k4 * f)
            comp4 = np.append(comp4, [l4])

        ret = np.log(pi1) * comp1
        ret += gamma1s * comp1
        ret += np.log(pi2) * comp2
        ret += gamma2s * comp2
        ret += np.log(pi3) * comp3
        ret += gamma3s * comp3
        ret += np.log(pi4) * comp4
        ret += gamma4s * comp4
        return np.sum(ret)

    def prob1(self, col, a1, a2, b, d, sq_sigma1, sq_sigma2):
        loc, scale = self.fs[col]
        if np.isnan(d):
            return 1 - (1 - b)**2
        else:
            k1 = hom_model.alfa1(a1, a2, b)
            di = norm.pdf(d, 0, sq_sigma1)

            k2 = hom_model.alfa2(a2, b)
            n1 = norm.pdf(d, 0, sq_sigma2)

            k3 = hom_model.alfa3(a1, a2, b)
            n2 = norm.pdf(d, 0, 2 * sq_sigma2)

            k4 = hom_model.alfa4(a1)
            f = norm.pdf(d, loc, scale)

            ret = k1 * di + k2 * f + k3 * n1 + k4 * n2

            return ret

    def prob2(self, col, b, d):
        loc, scale = self.fs[col]
        if np.isnan(d):
            return 1 - (1 - b)**2
        else:
            f = norm.pdf(d, loc, scale)
            return (1 - b)**2 * f

    def wjk(self, j, k):
        ret = 0
        (_, n) = self.data.shape
        for col in self.cols(Algorithm.HOM):
            ret += self.wjk_hom(j, k, col)
        for col in self.cols(Algorithm.DEV):
            ret += self.wjk_mixt(j, k, col)
        return ret

    def wjk_hom(self, j, k, col):
        x = self.data.loc[j, col]
        y = self.data.loc[k, col]
        b = self.bs[col]
        c = self.cs[col]
        beta = self.betas[col][x]

        if self.isna.loc[j, col] or self.isna.loc[k, col]:
            return 0
        elif x == y:
            return np.log2(1 - c * (1 - beta) * (1 - b)**(-2)) - np.log2(beta)
        else:
            return np.log2(c) - 2 * np.log2(1 - b)

    def wjk_mixt(self, j, k, col):
        a1 = self.a1s[col]
        a2 = self.a2s[col]
        b = self.bs[col]
        d = self.data.loc[j, col] - self.data.loc[k, col]
        eps = self.eps[col]
        s1 = self.sq_sigma1s[col]
        s2 = self.sq_sigma2s[col]
        fd = self.fs[col]
        if np.isnan(d):
            return 0
        else:
            fun1 = lambda x: self.prob1(col, a1, a2, b, x, s1, s2)
            fun2 = lambda x: self.prob2(col, b, x)
            i1, e1 = integrate.quad(fun1, d - eps, d + eps)
            i2, e2 = integrate.quad(fun2, d - eps, d + eps)
            if i2 == 0:
                print("Division by zero upcoming")
                return np.PINF
            return np.log2(i1 / i2)

    def calc_thres(self, train):
        """Calculate the threshold to classify a pair as duplicate.

        Args:
            train: dataframe containing the train data

        Returns:
            A float representing the threshold
        """
        try:
            train.shape
        except AttributeError as N:
            err_msg ="Incorrect use of function calc_thres:\n"
            print(err_msg, self.calc_thres.__doc__)
            return None
        
        groups = train.dropna()

        score_dupl = []
        for _, row in groups.iterrows():
            j = int(row[0])  # StringToInt to get i
            ks = [int(x)
                  for x in row[1].split(';')]  # [StringToInt] to get [j]
            for k in ks:
                score = self.wjk(j, k)
                if np.isfinite(score):
                    score_dupl.append(score)
        mr, sr = norm.fit(score_dupl)

        (n, _) = train.shape
        
        samp = zip(random.sample(list(self.data.index), k=int(0.25 * n)),
                   random.sample(list(self.data.index), k=int(0.25 * n)))
        score_samp = []
        for j, k in samp:
            score = self.wjk(j, k)
            if np.isfinite(score):
                score_samp.append(self.wjk(j, k))
                
        mu, su = norm.fit(score_samp)

        (n, _) = self.data.shape
        dupl = 0.05
        self.t = hom_model.solve_bayesrule(dupl, mr, sr, mu, su)

    def solve_bayesrule(dupl, mr, sr, mu, su):
        warr = 0.95
        foo = lambda x: warr * (dupl * norm.pdf(x, mr, sr) + (1 - dupl) * norm.
                                pdf(x, mu, su)) - dupl * norm.pdf(x, mr, sr)
        plot(np.linspace(-50, 50),
             [foo(x) for x in np.linspace(-50,50)])
        return optimize.newton(foo, 0)


if __name__ == "__main__":
    unittest.main()
