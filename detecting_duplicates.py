import itertools

import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm
from scipy.spatial.distance import pdist
import scipy.integrate as integrate

import json

# Test
import unittest
import random

# DEBUG
import pdb

def expand_grid(data_dict):
    """Builds a pandas dataframe by building the cartesian product of some values

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
    
    def test_blank_frequencies(self):
        df = pd.DataFrame({'A': [np.nan, 1.0, 1.0, 1.0],
                           'B': [np.nan, np.nan, 1.0, 1.0],
                           'C': [np.nan, np.nan, np.nan, 1.0],
                           'D': [np.nan, np.nan, np.nan, np.nan]})
        model = hom_model(df, None, [True]*4, None)
        bf = {'A': 0.25, 'B': 0.5, 'C': 0.75, 'D': 1.0}
        self.assertEqual(model.bs, bf,
                         'incorrect frequency of blank elements')

    def test_relative_frequencies(self):
        df = pd.DataFrame({'A': [1.0, 1.0, 1.0, 1.0],
                           'B': [1.0, 1.0, 2.0, 2.0],
                           'C': [1.0, 2.0, 2.0, 3.0],
                           'D': [1.0, 2.0, 3.0, 4.0]})
        model = hom_model(df, None, [True]*4, None)
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
        
    def test_calculate_differences(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0],
                           'B': [5.0, 5.0, 6.0, 7.0]})
        train = pd.DataFrame({'Issue_id': ["0", "1", "2", "3"],
                              'Duplicate': ["1", "0", "3", "2"]})
        model = hom_model(df, train, [False]*2, None)
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
        model = hom_model(df, train, [True]*2, None)
        dp = {'A': 4.0/3, 'B': 4.0/5}
        self.assertEqual(model.cs, dp,
                         'incorrect discordant pairs freq.')

    def test_saving_coefficients(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0],
                           'B': [5.0, 5.0, 6.0, 7.0]})
        train = pd.DataFrame({'Issue_id': ["0", "1", "2", "3"],
                              'Duplicate': ["1", "0", "3", "2"]})
        model = hom_model(df, train, [True]*2, None)

        path = './tests/test_saving_coeffs.json'
        model.save_coefficients(path)

        with open(path, 'r') as f:
            s = f.read().strip('\n')
            
        synth = json.dumps({"bs": model.bs,
                            "betas": model.betas,
                            "ds": {k: v.tolist() for k,v in model.ds},
                            "fs": model.fs,
                            "cs": model.cs, 
                            "a1s": model.a1s,
                            "a2s": model.a2s,
                            "var1s": model.var1s,
                            "var2s": model.var2s,
                            "eps": model.eps})
        
        self.assertEqual(s, synth,
                         'Incorrect save of coefficients')

    def test_initialize_from_file(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0],
                           'B': [5.0, 5.0, 6.0, 7.0]})
        train = pd.DataFrame({'Issue_id': ["0", "1", "2", "3"],
                              'Duplicate': ["1", "0", "3", "2"]})
        model = hom_model(df, train, [True]*2, None)

        path = './tests/test_saving_coeffs.json'
        model.save_coefficients(path)

        model_file = hom_model(path=path)

        values = [model.bs,
                  model.betas,
                  model.fs,
                  model.cs,
                  model.a1s,
                  model.a2s,
                  model.var1s,
                  model.var2s,
                  model.eps]

        values_file = [model_file.bs,
                       model_file.betas,
                       model_file.fs,
                       model_file.cs,
                       model_file.a1s,
                       model_file.a2s,
                       model_file.var1s,
                       model_file.var2s,
                       model_file.eps]

        self.assertEqual(values, values_file,
                         'Incorrect save of coefficients')
        
    # TODO: Resto de tests
    
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
    def __init__(self, *args, **kwargs):
        """Initialize hit or miss probabilistic model
        
        Args:
            data: dataframe with the data
            train: dataframe used to train the model
            algs: array specifying the algorithm used for each column
            eps: dictionary that stores possible deviations that the data inside a column can have
            
        Returns:
            A hom model for estimating how similar two rows of the data are
        """
        if 'path' in kwargs:
            self.file_constructor(kwargs['path'])
        else:
            data = args[0]
            train = args[1]
            algs = args[2]
            eps = args[3]
            self.data_constructor(data, train, algs, eps)
        
    def data_constructor(self, data, train, algs, eps):
        self.data = data
        self.algs = algs
        self.eps = eps
        self.bs = self.blank_freq()
        self.betas = self.rel_freq()
        self.ds = self.differences(train)
        self.fs = self.norm_fits()
        self.cs = self.dp_freq(train)
        self.a1s, self.a2s, self.var1s, self.var2s = self.calculate_frequencies()
        self.eps = eps

    def file_constructor(self,path):
        with open(path, 'r') as f:
            params = json.loads(f.read().strip('\n'))
        self.bs = params["bs"]
        self.betas = {k: {float(kk):vv for kk, vv in v.items()} for k,v in params["betas"].items()}
        self.ds = params["ds"]
        self.fs = params["fs"]
        self.cs = params["cs"]
        self.a1s = params["a1s"]
        self.a2s = params["a2s"]
        self.var1s = params["var1s"]
        self.var2s = params["var2s"]
        self.eps = params["eps"]

    def blank_freq(self):
        """Returns the blank frequencies for each column"""
        (n, _) = self.data.shape
        calc_bf = lambda x: 1 - float(x)/ n
        return self.data.count().apply(calc_bf).to_dict()

    def cols(self, alg):
        """Returns the columns where a certain algorithm is executed"""
        it = zip(self.data.columns, self.algs)
        return (col for col, a in it if a == alg)

    def get_fs(self, col):
        df = self.data.loc[:,col].dropna()
        (n, _) = self.data.shape  # nxm -> n
        norm = lambda x: float(x) / n
        if not df.empty:
            return self.data.groupby(col).size().apply(norm).to_dict()
        else:
            return {}
    
    def rel_freq(self):
        """Returns the frequency for each value inside each column"""
        cols = self.cols(True)
        return {col: self.get_fs(col) for col in cols}

    def get_ds(self, index_sel, col):
        """Calculates elementwise differences for a specific column 
        of a subselection of the data. This subselection is indexed by
        a list of indexes.
        """
        xs = self.data.loc[index_sel, col].dropna().to_numpy()
        return pdist(xs.reshape(len(xs), 1), lambda x, y: x-y)
    

    # Utilizar el resultado de esta funcion en norm_fits
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
            print err_msg + self.differences.__doc__
            return None

        train_data = train.dropna().loc[:, 'Issue_id'].apply(int)
        cols = self.cols(False)
        return {col: self.get_ds(train_data, col) for col in cols}

    def norm_fits(self):
        """Fits a normal distribution to the differences of the elements in a numerical column"""
        if self.ds:
            return {k: norm.fit(self.ds[k]) for k in self.ds}
        return None

    # TODO: menos lineas de codigo
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
            print err_msg + self.dp_freq.__doc__
            return None

        groups = train.dropna()  # Drop rows with invalid values
        (n, _) = groups.shape

        cols = self.cols(True)
        ret = {col: 0 for col in cols}
        for _, row in groups.iterrows():
            i = int(row[0])  # StringToInt to get i
            js = [int(x)
                  for x in row[1].split(';')]  # [StringToInt] to get [j]
            for j in js:
                for col in self.cols(True):
                    if self.data.loc[i, col] != self.data.loc[j, col]:
                        # Notice that every discordant pair DP is
                        # added twice. First when chosen by (i,j)
                        # and second when chosen by (j,i)
                        ret[col] += 1  # Adds a new discordant pair
        for col in self.cols(True):
            s = 0
            for key in self.betas[col]:
                s += (self.betas[col][key])**2
            # The two is due to the fact that DPs are added twice
            ret[col] = float(ret[col])/n * (1 - s)
        return ret

    def calculate_frequencies(self):
        """Calculate estimates for HOM DEV algorithm"""
        a1s, a2s, s1s, s2s = dict(), dict(), dict(), dict()
        for c in self.cols(False):
            a1s[c], a2s[c], s1s[c], s2s[c] = self.hom_mix_fitting(c)
        return a1s, a2s, s1s, s2s

    def hom_mix_fitting(self, col):
        """Computes the a1, a2 and variance for a column of the dataset
        
        Args:
            col: The column where the hit or miss deviation algorithm 
            calculates constants
            
        Returns:
            A triple with the EM algorithm output for a1, a2 and variance
        """
        def shake_val(x):
            """Auxiliary function to escape a cycle"""
            return x + 100 * random.random()
        # First estimation of sigmas MAYBE: sq_sigma -> var
        var2 = np.var(self.ds[col])
        var1 = 0.001*var2

        # Initial probability guess
        b = self.bs[col]
        a1 = (1-b)*random.random()
        a2 = (1 - b - a1)*random.random()

        d = np.linalg.norm([a1, a2, var1, var2])
        eps = 0.001
        while d > eps:
            a1_, a2_, var1_, var2_ = a1, a2, var1, var2
            a1, a2, var1, var2 = self.hom_mix_fitting_loop(col, a1, a2, var1, var2)
            v = [a1 - a1_, a2 - a2_, var1 - var1_, var2 - var2_]
            d_, d = d, np.linalg.norm(v)
            if d_ == d:
                a1 = shake_val(a1)
                a2 = shake_val(a2)
                var1 = shake_val(var1)
                var2 = shake_val(var2)
        return a1, a2, var1, var2

    def hom_mix_fitting_loop(self, col, a1, a2, var1, sq_sigma2):
        """Calculate constants a1, a2, var1, var2 by means of an EM algorithm"""
        b = self.bs[col]
        ds = self.ds[col]

        gamma1s = np.array([])
        gamma2s = np.array([])
        gamma3s = np.array([])
        gamma4s = np.array([])

        for d in ds:
            gamma1, gamma2, gamma3, gamma4 = self.gammas(
                col, d, a1, a2, b, var1, sq_sigma2)
            gamma1s = np.append(gamma1s, [gamma1])
            gamma2s = np.append(gamma2s, [gamma2])
            gamma3s = np.append(gamma3s, [gamma3])
            gamma4s = np.append(gamma4s, [gamma4])

        num1 = np.sum(gamma1s * ds**2)
        den1 = np.sum(gamma1s)
        var1 = num1 / den1

        num2 = np.sum((gamma3s + gamma4s * 0.5) * ds**2)
        den2 = np.sum(gamma3s + gamma4s)
        sq_sigma2 = num2 / den2

        a1, a2 = self.update_as(col, a1, a2, b, ds, var1, sq_sigma2, gamma1s, gamma2s,
                                gamma3s, gamma4s)

        return a1, a2, var1, sq_sigma2

    def alfa1(self,a1, a2, b):
        """Probability of the first distribution in the mixture model"""
        return float((1 - a1 - a2 - b)**2)

    def alfa2(self,a2, b):
        """Probability of the second distribution in the mixture model"""
        return float(a2 * (2 - 2 * b - a2))

    def alfa3(self,a1, a2, b):
        """Probability of the third distribution in the mixture model"""
        return float(2 * a1 * (1 - a1 - a2 - b))

    def alfa4(self,a1):
        """Probability of the second distribution in the mixture model"""
        return float(a1**2)

    def gammas(self, col, di, a1, a2, b, var1, var2):
        """Calculates responsability for a specific difference within the EM algorithm"""
        alfa1 = self.alfa1(a1, a2, b)
        alfa2 = self.alfa2(a2, b)
        alfa3 = self.alfa3(a1, a2, b)
        alfa4 = self.alfa4(a1)

        norm1 = norm.pdf(di, 0, var1)

        loc, scale = self.fs[col]
        f = norm.pdf(di, loc, scale)

        norm2 = norm.pdf(di, 0, var2)

        norm3 = norm.pdf(di, 0, 2 * var2)

        gamma1 = alfa1 * norm1
        gamma2 = alfa2 * f
        gamma3 = alfa3 * norm2
        gamma4 = alfa4 * norm3

        den = gamma1 + gamma2 + gamma3 + gamma4

        return gamma1 / den, gamma2 / den, gamma3 / den, gamma4 / den

    def update_as(self, col, a1, a2, b, ds, var1, var2, gamma1s, gamma2s,
                  gamma3s, gamma4s):
        """Updates probabilities a1 and a2 by maximizing the expected likelihood"""
        x0 = [a1, a2]
        cons = ({'type': 'ineq', 'fun': lambda x: 1 - b - x[0] - x[1]})
        bnds = ((0, 1), (0, 1))
        fun = lambda x: -self.expected_likelihood(
            col, x, b, ds, var1, var2, gamma1s, gamma2s, gamma3s, gamma4s)
        res = optimize.minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
        print "The expected likelihood is: {}".format(-fun(res.x))
        return res.x[0], res.x[1]

    def expected_likelihood(self, col, x, b, ds, var1, var2, gamma1s, gamma2s,
                            gamma3s, gamma4s):
        """Returns expected likelihood for our mixture model"""
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

        k1 = self.alfa1(a1, a2, b)
        k2 = self.alfa2(a2, b)
        k3 = self.alfa3(a1, a2, b)
        k4 = self.alfa4(a1)
        for d in ds:
            di = norm.pdf(d, 0, var1)
            l1 = np.log2(k1 * di)
            comp1 = np.append(comp1, [l1])

            n1 = norm.pdf(d, 0, var2)
            l2 = np.log2(k2 * n1)
            comp2 = np.append(comp2, [l2])

            n2 = norm.pdf(d, 0, 2 * var2)
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

    def prob1(self, col, a1, a2, b, d, var1, var2):
        """Probability of a difference occuring withing a column with a HOM model"""
        loc, scale = self.fs[col]
        if np.isnan(d):
            return 1 - (1 - b)**2
        else:
            k1 = self.alfa1(a1, a2, b)
            di = norm.pdf(d, 0, var1)

            k2 = self.alfa2(a2, b)
            n1 = norm.pdf(d, 0, var2)

            k3 = self.alfa3(a1, a2, b)
            n2 = norm.pdf(d, 0, 2 * var2)

            k4 = self.alfa4(a1)
            f = norm.pdf(d, loc, scale)

            ret = k1 * di + k2 * f + k3 * n1 + k4 * n2

            return ret

    def prob2(self, col, b, d):
        """Probability of a difference occuring withing a column with a deviated HOM model"""
        loc, scale = self.fs[col]
        if np.isnan(d):
            return 1 - (1 - b)**2
        else:
            f = norm.pdf(d, loc, scale)
            return (1 - b)**2 * f

    def wjk(self, j, k):
        """Calculates log likelihood ratio for two rows j and k   

        j: index of the first row
        k: index of the second row


        returns a log likelihood ratio the bigger the ratio the 
        more likely it is for two rows to be equal
        """
        ret = 0
        (_, n) = self.data.shape
        for col in self.cols(True):
            ret += self.wjk_hom(j, k, col)
        for col in self.cols(False):
            ret += self.wjk_mixt(j, k, col)
        return ret

    def wjk_hom(self, j, k, col):
        """Calculates log likelihood ratio for two rows j 
        and k in a HOM column
        """
        x = self.data.loc[j, col]
        y = self.data.loc[k, col]
        b = self.bs[col]
        c = self.cs[col]
        beta = self.betas[col][x]

        if pd.isna(x) or pd.isna(y):
            return 0
        elif x == y:
            return np.log2(1 - c * (1 - beta) * (1 - b)**(-2)) - np.log2(beta)
        else:
            return np.log2(c) - 2 * np.log2(1 - b)

    def wjk_mixt(self, j, k, col):
        """Calculates log likelihood ratio for two rows j and k
        in a deviated HOM column
        """
        a1 = self.a1s[col]
        a2 = self.a2s[col]
        b = self.bs[col]
        d = self.data.loc[j, col] - self.data.loc[k, col]
        eps = self.eps[col]
        s1 = self.var1s[col]
        s2 = self.var2s[col]
        fd = self.fs[col]
        if np.isnan(d):
            return 0
        else:
            fun1 = lambda x: self.prob1(col, a1, a2, b, x, s1, s2)
            fun2 = lambda x: self.prob2(col, b, x)
            i1, e1 = integrate.quad(fun1, d - eps, d + eps)
            i2, e2 = integrate.quad(fun2, d - eps, d + eps)
            if i2 == 0:
                print "Division by zero upcoming"
                return np.PINF
            return np.log2(float(i1) / i2)

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
            print err_msg + self.calc_thres.__doc__
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

        dupl = 0.005
        self.t = self.solve_bayesrule(dupl, mr, sr, mu, su)

    def solve_bayesrule(self,dupl, mr, sr, mu, su):
        """Returns the score that gives at least 0.95 probability 
        of two rows being duplicates considered that score
        """
        warr = 0.95
        foo = lambda x: warr * (dupl * norm.pdf(x, mr, sr) + (1 - dupl) * norm.
                                pdf(x, mu, su)) - dupl * norm.pdf(x, mr, sr)
        # TODO: search for optunity function to find zeros
        while True:
            try:
                x0 = random.random()
                return optimize.newton(foo, x0)
            except:
                x0 = random.random()

    def save_coefficients(self, path):
        ds = {k: v.tolist() for k,v in self.ds.items()}
        values = {"bs": self.bs,
                  "betas": self.betas,
                  "ds": ds,
                  "fs": self.fs,
                  "cs": self.cs,
                  "a1s": self.a1s,
                  "a2s": self.a2s,
                  "var1s": self.var1s,
                  "var2s": self.var2s,
                  "eps": self.eps}
        st = json.dumps(values)
        with open(path, 'w') as f:
            f.write(st)

if __name__ == "__main__":
    unittest.main()
