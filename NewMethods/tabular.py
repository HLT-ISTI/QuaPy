import numpy as np
import itertools
from scipy.stats import ttest_ind_from_stats, wilcoxon


class Table:
    VALID_TESTS = [None, "wilcoxon", "ttest"]

    def __init__(self, benchmarks, methods, lower_is_better=True, ttest='ttest', prec_mean=3,
                 clean_zero=False, show_std=False, prec_std=3, average=True, missing=None, missing_str='--', color=True):
        assert ttest in self.VALID_TESTS, f'unknown test, valid are {self.VALID_TESTS}'

        self.benchmarks = np.asarray(benchmarks)
        self.benchmark_index = {row:i for i, row in enumerate(benchmarks)}

        self.methods = np.asarray(methods)
        self.method_index = {col:j for j, col in enumerate(methods)}

        self.map = {}  
        # keyed (#rows,#cols)-ndarrays holding computations from self.map['values']
        self._addmap('values', dtype=object)
        self.lower_is_better = lower_is_better
        self.ttest = ttest
        self.prec_mean = prec_mean
        self.clean_zero = clean_zero
        self.show_std = show_std
        self.prec_std = prec_std
        self.add_average = average
        self.missing = missing
        self.missing_str = missing_str
        self.color = color
        
        self.touch()

    @property
    def nbenchmarks(self):
        return len(self.benchmarks)

    @property
    def nmethods(self):
        return len(self.methods)

    def touch(self):
        self._modif = True

    def update(self):
        if self._modif:
            self.compute()

    def _getfilled(self):
        return np.argwhere(self.map['fill'])

    @property
    def values(self):
        return self.map['values']

    def _indexes(self):
        return itertools.product(range(self.nbenchmarks), range(self.nmethods))

    def _addmap(self, map, dtype, func=None):
        self.map[map] = np.empty((self.nbenchmarks, self.nmethods), dtype=dtype)
        if func is None:
            return
        m = self.map[map]
        f = func
        indexes = self._indexes() if map == 'fill' else self._getfilled()
        for i, j in indexes:
            m[i, j] = f(self.values[i, j])

    def _addrank(self):
        for i in range(self.nbenchmarks):
            filled_cols_idx = np.argwhere(self.map['fill'][i]).flatten()
            col_means = [self.map['mean'][i,j] for j in filled_cols_idx]
            ranked_cols_idx = filled_cols_idx[np.argsort(col_means)]
            if not self.lower_is_better:
                ranked_cols_idx = ranked_cols_idx[::-1]
            self.map['rank'][i, ranked_cols_idx] = np.arange(1, len(filled_cols_idx)+1)
            
    def _addcolor(self):
        for i in range(self.nbenchmarks):
            filled_cols_idx = np.argwhere(self.map['fill'][i]).flatten()
            if filled_cols_idx.size==0:
                continue
            col_means = [self.map['mean'][i,j] for j in filled_cols_idx]
            minval = min(col_means)
            maxval = max(col_means)
            for col_idx in filled_cols_idx:
                val = self.map['mean'][i,col_idx]
                norm = (maxval - minval)
                if norm > 0:
                    normval = (val - minval) / norm
                else:
                    normval = 0.5
                if self.lower_is_better:
                    normval = 1 - normval
                self.map['color'][i, col_idx] = color_red2green_01(normval)

    def _run_ttest(self, row, col1, col2):
        mean1 = self.map['mean'][row, col1]
        std1 = self.map['std'][row, col1]
        nobs1 = self.map['nobs'][row, col1]
        mean2 = self.map['mean'][row, col2]
        std2 = self.map['std'][row, col2]
        nobs2 = self.map['nobs'][row, col2]
        _, p_val = ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
        return p_val

    def _run_wilcoxon(self, row, col1, col2):
        values1 = self.map['values'][row, col1]
        values2 = self.map['values'][row, col2]
        _, p_val = wilcoxon(values1, values2)
        return p_val

    def _add_statistical_test(self):
        if self.ttest is None:
            return
        self.some_similar = [False]*self.nmethods
        for i in range(self.nbenchmarks):
            filled_cols_idx = np.argwhere(self.map['fill'][i]).flatten()
            if len(filled_cols_idx) <= 1:
                continue
            col_means = [self.map['mean'][i,j] for j in filled_cols_idx]
            best_pos = filled_cols_idx[np.argmin(col_means)]

            for j in filled_cols_idx:
                if j==best_pos:
                    continue
                if self.ttest == 'ttest':
                    p_val = self._run_ttest(i, best_pos, j)
                else:
                    p_val = self._run_wilcoxon(i, best_pos, j)

                pval_outcome = pval_interpretation(p_val)
                self.map['ttest'][i, j] = pval_outcome
                if pval_outcome != 'Diff':
                    self.some_similar[j] = True

    def compute(self):
        self._addmap('fill', dtype=bool, func=lambda x: x is not None)
        self._addmap('mean', dtype=float, func=np.mean)
        self._addmap('std', dtype=float, func=np.std)
        self._addmap('nobs', dtype=float, func=len)
        self._addmap('rank', dtype=int, func=None)
        self._addmap('color', dtype=object, func=None)
        self._addmap('ttest', dtype=object, func=None)
        self._addmap('latex', dtype=object, func=None)
        self._addrank()
        self._addcolor()
        self._add_statistical_test()
        if self.add_average:
            self._addave()
        self._modif = False

    def _is_column_full(self, col):
        return all(self.map['fill'][:, self.method_index[col]])

    def _addave(self):
        ave = Table(['ave'], self.methods, lower_is_better=self.lower_is_better, ttest=self.ttest, average=False,
                    missing=self.missing, missing_str=self.missing_str)
        for col in self.methods:
            values = None
            if self._is_column_full(col):
                if self.ttest == 'ttest':
                    values = np.asarray(self.map['mean'][:, self.method_index[col]])
                else:  # wilcoxon
                    values = np.concatenate(self.values[:, self.method_index[col]])
            ave.add('ave', col, values)
        self.average = ave

    def add(self, benchmark, method, values):
        if values is not None:
            values = np.asarray(values)
            if values.ndim==0:
                values = values.flatten()
        rid, cid = self._coordinates(benchmark, method)
        self.map['values'][rid, cid] = values
        self.touch()

    def get(self, benchmark, method, attr='mean'):
        self.update()
        assert attr in self.map, f'unknwon attribute {attr}'
        rid, cid = self._coordinates(benchmark, method)
        if self.map['fill'][rid, cid]:
            v = self.map[attr][rid, cid]
            if v is None or (isinstance(v,float) and np.isnan(v)):
                return self.missing
            return v
        else:
            return self.missing

    def _coordinates(self, benchmark, method):
        assert benchmark in self.benchmark_index, f'benchmark {benchmark} out of range'
        assert method in self.method_index, f'method {method} out of range'
        rid = self.benchmark_index[benchmark]
        cid = self.method_index[method]
        return rid, cid

    def get_average(self, method, attr='mean'):
        self.update()
        if self.add_average:
            return self.average.get('ave', method, attr=attr)
        return None

    def get_color(self, benchmark, method):
        color = self.get(benchmark, method, attr='color')
        if color is None:
            return ''
        return color

    def latex(self, benchmark, method):
        self.update()
        i,j = self._coordinates(benchmark, method)
        if self.map['fill'][i,j] == False:
            return self.missing_str

        mean = self.map['mean'][i,j]
        l = f" {mean:.{self.prec_mean}f}"
        if self.clean_zero:
            l = l.replace(' 0.', '.')

        isbest = self.map['rank'][i,j] == 1
        if isbest:
            l = "\\textbf{"+l.strip()+"}"

        stat = ''
        if self.ttest is not None and self.some_similar[j]:
            test_label = self.map['ttest'][i,j]
            if test_label == 'Sim':
                stat = '^{\dag\phantom{\dag}}'
            elif test_label == 'Same':
                stat = '^{\ddag}'
            elif isbest or test_label == 'Diff':
                stat = '^{\phantom{\ddag}}'

        std = ''
        if self.show_std:
            std = self.map['std'][i,j]
            std = f" {std:.{self.prec_std}f}"
            if self.clean_zero:
                std = std.replace(' 0.', '.')
            std = f" \pm {std:{self.prec_std}}"

        if stat!='' or std!='':
            l = f'{l}${stat}{std}$'

        if self.color:
            l += ' ' + self.map['color'][i,j]

        return l

    def latexTabular(self, benchmark_replace={}, method_replace={}, average=True):
        tab = ' & '
        tab += ' & '.join([method_replace.get(col, col) for col in self.methods])
        tab += ' \\\\\hline\n'
        for row in self.benchmarks:
            rowname = benchmark_replace.get(row, row)
            tab += rowname + ' & '
            tab += self.latexRow(row)

        if average:
            tab += '\hline\n'
            tab += 'Average & '
            tab += self.latexAverage()
        return tab

    def latexRow(self, benchmark, endl='\\\\\hline\n'):
        s = [self.latex(benchmark, col) for col in self.methods]
        s = ' & '.join(s)
        s += ' ' + endl
        return s

    def latexAverage(self, endl='\\\\\hline\n'):
        if self.add_average:
            return self.average.latexRow('ave', endl=endl)

    def getRankTable(self):
        t = Table(benchmarks=self.benchmarks, methods=self.methods, prec_mean=0, average=True)
        for rid, cid in self._getfilled():
            row = self.benchmarks[rid]
            col = self.methods[cid]
            t.add(row, col, self.get(row, col, 'rank'))
        t.compute()
        return t

    def dropMethods(self, methods):
        drop_index = [self.method_index[m] for m in methods]
        new_methods = np.delete(self.methods, drop_index)
        new_index = {col:j for j, col in enumerate(new_methods)}

        self.map['values'] = self.values[:,np.asarray([self.method_index[m] for m in new_methods], dtype=int)]
        self.methods = new_methods
        self.method_index = new_index
        self.touch()


def pval_interpretation(p_val):
    if 0.005 >= p_val:
        return 'Diff'
    elif 0.05 >= p_val > 0.005:
        return 'Sim'
    elif p_val > 0.05:
        return 'Same'


def color_red2green_01(val, maxtone=50):
    if np.isnan(val): return None
    assert 0 <= val <= 1, f'val {val} out of range [0,1]'

    # rescale to [-1,1]
    val = val * 2 - 1
    if val < 0:
        color = 'red'
        tone = maxtone * (-val)
    else:
        color = 'green'
        tone = maxtone * val
    return '\cellcolor{' + color + f'!{int(tone)}' + '}'

