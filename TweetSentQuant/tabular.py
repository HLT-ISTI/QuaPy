import numpy as np
import itertools
from scipy.stats import ttest_ind_from_stats, wilcoxon


class Table:
    VALID_TESTS = [None, "wilcoxon", "ttest"]

    def __init__(self, rows, cols, addfunc, lower_is_better=True, ttest='ttest', prec_mean=3, clean_zero=False,
                 show_std=False, prec_std=3):
        assert ttest in self.VALID_TESTS, f'unknown test, valid are {self.VALID_TESTS}'
        self.rows = np.asarray(rows)
        self.row_index = {row:i for i,row in enumerate(rows)}
        self.cols = np.asarray(cols)
        self.col_index = {col:j for j,col in enumerate(cols)}
        self.map = {}
        self.mfunc = {}
        self.rarr = {}
        self.carr = {}
        self._addmap('values', dtype=object)
        self._addmap('fill', dtype=bool, func=lambda x: x is not None)
        self._addmap('mean', dtype=float, func=np.mean)
        self._addmap('std', dtype=float, func=np.std)
        self._addmap('nobs', dtype=float, func=len)
        self._addmap('rank', dtype=int, func=None)
        self._addmap('color', dtype=object, func=None)
        self._addmap('ttest', dtype=object, func=None)
        self._addrarr('mean', dtype=float, func=np.mean, argmap='mean')
        self._addrarr('min', dtype=float, func=np.min, argmap='mean')
        self._addrarr('max', dtype=float, func=np.max, argmap='mean')
        self._addcarr('mean', dtype=float, func=np.mean, argmap='mean')
        self._addcarr('rank-mean', dtype=float, func=np.mean, argmap='rank')
        if self.nrows>1:
            self._col_ttest = Table(['ttest'], cols, _merge, lower_is_better, ttest)
        else:
            self._col_ttest = None
        self.addfunc = addfunc
        self.lower_is_better = lower_is_better
        self.ttest = ttest
        self.prec_mean = prec_mean
        self.clean_zero = clean_zero
        self.show_std = show_std
        self.prec_std = prec_std
        self.touch()

    @property
    def nrows(self):
        return len(self.rows)

    @property
    def ncols(self):
        return len(self.cols)

    def touch(self):
        self.modif = True

    def update(self):
        if self.modif:
            self.compute()

    def _addmap(self, map, dtype, func=None):
        self.map[map] = np.empty((self.nrows, self.ncols), dtype=dtype)
        self.mfunc[map] = func
        self.touch()

    def _addrarr(self, rarr, dtype, func=np.mean, argmap='mean'):
        self.rarr[rarr] = {
            'arr': np.empty(self.ncols, dtype=dtype),
            'func': func,
            'argmap': argmap
        }
        self.touch()

    def _addcarr(self, carr, dtype, func=np.mean, argmap='mean'):
        self.carr[carr] = {
            'arr': np.empty(self.nrows, dtype=dtype),
            'func': func,
            'argmap': argmap
        }
        self.touch()

    def _getfilled(self):
        return np.argwhere(self.map['fill'])

    @property
    def values(self):
        return self.map['values']

    def _indexes(self):
        return itertools.product(range(self.nrows), range(self.ncols))

    def _runmap(self, map):
        m = self.map[map]
        f = self.mfunc[map]
        if f is None:
            return
        indexes = self._indexes() if map == 'fill' else self._getfilled()
        for i,j in indexes:
            m[i,j] = f(self.values[i,j])

    def _runrarr(self, rarr):
        dic = self.rarr[rarr]
        arr, f, map = dic['arr'], dic['func'], dic['argmap']
        for col, cid in self.col_index.items():
            if all(self.map['fill'][:, cid]):
                arr[cid] = f(self.map[map][:, cid])
            else:
                arr[cid] = None

    def _runcarr(self, carr):
        dic = self.carr[carr]
        arr, f, map = dic['arr'], dic['func'], dic['argmap']
        for row, rid in self.row_index.items():
            if all(self.map['fill'][rid, :]):
                arr[rid] = f(self.map[map][rid, :])
            else:
                arr[rid] = None

    def _runrank(self):
        for i in range(self.nrows):
            filled_cols_idx = np.argwhere(self.map['fill'][i]).flatten()
            col_means = [self.map['mean'][i,j] for j in filled_cols_idx]
            ranked_cols_idx = filled_cols_idx[np.argsort(col_means)]
            if not self.lower_is_better:
                ranked_cols_idx = ranked_cols_idx[::-1]
            self.map['rank'][i, ranked_cols_idx] = np.arange(1, len(filled_cols_idx)+1)

    def _runcolor(self):
        for i in range(self.nrows):
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

    def _runttest(self):
        if self.ttest is None:
            return
        self.some_similar = False
        for i in range(self.nrows):
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
                    self.some_similar = True

    def get_col_average(self, col, arr='mean'):
        self.update()
        cid = self.col_index[col]
        return self.rarr[arr]['arr'][cid]

    def _map_list(self):
        maps = list(self.map.keys())
        maps.remove('fill')
        maps.remove('values')
        maps.remove('color')
        maps.remove('ttest')
        return ['fill'] + maps

    def compute(self):
        for map in self._map_list():
            self._runmap(map)
        self._runrank()
        self._runcolor()
        self._runttest()
        for arr in self.rarr.keys():
            self._runrarr(arr)
        for arr in self.carr.keys():
            self._runcarr(arr)
        if self._col_ttest != None:
            for col in self.cols:
                self._col_ttest.add('ttest', col, self.col_index[col], self.map['fill'], self.values, self.map['mean'], self.ttest)
                self._col_ttest.compute()
        self.modif = False

    def add(self, row, col, *args, **kwargs):
        print(row, col, args, kwargs)
        values = self.addfunc(row, col, *args, **kwargs)
        # if values is None:
        #     raise ValueError(f'addfunc returned None for row={row} col={col}')
        rid, cid = self.coord(row, col)
        self.map['values'][rid, cid] = values
        self.touch()

    def get(self, row, col, attr='mean'):
        assert attr in self.map, f'unknwon attribute {attr}'
        self.update()
        rid, cid = self.coord(row, col)
        if self.map['fill'][rid, cid]:
            return self.map[attr][rid, cid]

    def coord(self, row, col):
        assert row in self.row_index, f'row {row} out of range'
        assert col in self.col_index, f'col {col} out of range'
        rid = self.row_index[row]
        cid = self.col_index[col]
        return rid, cid

    def get_col_table(self):
        return self._col_ttest

    def get_color(self, row, col):
        color = self.get(row, col, attr='color')
        if color is None:
            return ''
        return color

    def latex(self, row, col, missing='--', color=True):
        self.update()
        i,j = self.coord(row, col)
        if self.map['fill'][i,j] == False:
            return missing

        mean = self.map['mean'][i,j]
        l = f" {mean:.{self.prec_mean}f}"
        if self.clean_zero:
            l = l.replace(' 0.', '.')

        isbest = self.map['rank'][i,j] == 1

        if isbest:
            l = "\\textbf{"+l+"}"
        else:
            if self.ttest is not None and self.some_similar:
                test_label = self.map['ttest'][i,j]
                if test_label == 'Sim':
                    l += '^{\dag\phantom{\dag}}'
                elif test_label == 'Same':
                    l += '^{\ddag}'
                elif test_label == 'Diff':
                    l += '^{\phantom{\ddag}}'

        if self.show_std:
            std = self.map['std'][i,j]
            std = f" {std:.{self.prec_std}f}"
            if self.clean_zero:
                std = std.replace(' 0.', '.')
            l += f" \pm {std}"

        l = f'$ {l} $'
        if color:
            l += ' ' + self.map['color'][i,j]

        return l

    def latextabular(self, missing='--', color=True, rowreplace={}, colreplace={}, average=True):
        tab = ' & '
        tab += ' & '.join([colreplace.get(col, col) for col in self.cols])
        tab += ' \\\\\hline\n'
        for row in self.rows:
            rowname = rowreplace.get(row, row)
            tab += rowname + ' & '
            tab += self.latexrow(row, missing, color)
            tab += ' \\\\\hline\n'

        if average:
            tab += 'Average & '
            tab += self.latexave(missing, color)
            tab += ' \\\\\hline\n'
        return tab


    def latexrow(self, row, missing='--', color=True):
        s = [self.latex(row, col, missing=missing, color=color) for col in self.cols]
        s = ' & '.join(s)
        return s

    def latexave(self, missing='--', color=True):
        return self._col_ttest.latexrow('ttest')

    def get_rank_table(self):
        t = Table(rows=self.rows, cols=self.cols, addfunc=_getrank, ttest=None, prec_mean=0)
        for row, col in self._getfilled():
            t.add(self.rows[row], self.cols[col], row, col, self.map['rank'])
        return t

def _getrank(row, col, rowid, colid, rank):
    return [rank[rowid, colid]]

def _merge(unused, col, colidx, fill, values, means, ttest):
    if all(fill[:,colidx]):
        nrows = values.shape[0]
        if ttest=='ttest':
            values = np.asarray(means[:, colidx])
        else:  # wilcoxon
            values = [values[i, colidx] for i in range(nrows)]
            values = np.concatenate(values)
        return values
    else:
        return None

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

#
# def addfunc(m,d, mean, size):
#     return np.random.rand(size)+mean
#
# t = Table(rows = ['M1', 'M2', 'M3'], cols=['D1', 'D2', 'D3', 'D4'], addfunc=addfunc, ttest='wilcoxon')
# t.add('M1','D1', mean=0.5, size=100)
# t.add('M1','D2', mean=0.5, size=100)
# t.add('M2','D1', mean=0.2, size=100)
# t.add('M2','D2', mean=0.1, size=100)
# t.add('M2','D3', mean=0.7, size=100)
# t.add('M2','D4', mean=0.3, size=100)
# t.add('M3','D1', mean=0.9, size=100)
# t.add('M3','D2', mean=0, size=100)
#
# print(t.latextabular())
#
# print('rank')
# print(t.get_rank_table().latextabular())
