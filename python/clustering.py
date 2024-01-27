# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

"""
Streaming clustering models.
"""

import numpy as np
from dSalmon import swig as dSalmon_cpp
# from dSalmon import projection
from dSalmon.util import sanitizeData, sanitizeTimes, lookupDistance

class Clustering(object):
    """
    Base class for streaming clustering models.
    """

    def _init_model(self, p):
        pass

    def get_params(self, deep=True):
        """
        Return the used algorithm parameters as a dictionary.

        Parameters
        ----------
        deep: bool, default=True
            Ignored. Only for compatibility with scikit-learn.

        Returns
        -------
        params: dict
            Dictionary of parameters.
        """
        return self.params

    def set_params(self, **params):
        """
        Reset the model and set the parameters according to the
        supplied dictionary.

        Parameters
        ----------
        **params: dict
            Dictionary of parameters.
        """
        p = self.params.copy()
        for key in params:
            assert key in p, 'Unknown parameter: %s' % key
            p[key] = params[key]
        self._init_model(p)

    def fit_predict(self, X, times=None):
        """
        Process the next chunk of data and perform clustering.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.

        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample.
        """
        # In most cases, fitting isn't any faster than additionally
        # performing outlier scoring. We override this method only
        # when it yields faster processing.
        self.fit_predict(X, times)

    def _process_data(self, data):
        data = sanitizeData(data, self.params['float_type'])
        assert self.dimension == -1 or data.shape[1] == self.dimension
        self.dimension = data.shape[1]
        return data

    def _process_times(self, data, times):
        times = sanitizeTimes(times, data.shape[0], self.last_time, self.params['float_type'])
        self.last_time = times[-1]
        return times

class SDOcluststream(Clustering):
    """
    Streaming clustering based on Sparse Data Observers :cite:p:`Hartl2019`.

    Parameters
    ----------
    k: int
        Number of observers to use.

    T: int
        Characteristic time for the model.
        Increasing T makes the model adjust slower, decreasing T
        makes it adjust quicker.

    qv: float, optional (default=0.3)
        Ratio of unused observers due to model cleaning.

    x: int (default=6)
        Number of nearest observers to consider for clustering.

    metric: string
        Which distance metric to use. Currently supported metrics
        include 'chebyshev', 'cityblock', 'euclidean', and 'minkowsi'.

    metric_params: dict
        Parameters passed to the metric. Minkowsi distance requires
        setting an integer `p` parameter.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.

    seed: int (default=0)
        Random seed to use.

    zeta: float, optional (default=0.6)
        A new parameter.

    chi_min: int, optional (default=8)
        A new parameter.

    chi_prop: float, optional (default=0.05)
        A new parameter.

    e: int, optional (default=3)
        A new parameter.
    """
    
    def __init__(self, k, T, qv=0.3, x=6, metric='euclidean', metric_params=None,
                 float_type=np.float64, seed=0, return_sampling=False, zeta=0.6, chi_min=8, chi_prop=0.05, e=3):
        self.params = {k: v for k, v in locals().items() if k != 'self'}
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert 0 <= p['qv'] < 1, 'qv must be in [0,1)'
        assert p['x'] > 0, 'x must be > 0'
        assert p['k'] > 0, 'k must be > 0'
        assert p['T'] > 0, 'T must be > 0'
        assert 0 <= p['zeta'] < 1, 'zeta must be in [0,1)'
        assert p['chi_min'] > 0, 'chi_min must be > 0'
        assert 0 <= p['chi_prop'] < 1, 'chi_prop must be in [0,1)'
        assert p['e'] > 0, 'e must be > 0'

        # Map the Python metric name to the C++ distance function
        distance_function = lookupDistance(p['metric'], p['float_type'], **(p['metric_params'] or {}))
        
        # Create an instance of the C++ SDOcluststream class
        cpp_obj = {
            np.float32: dSalmon_cpp.SDOcluststream32,
            np.float64: dSalmon_cpp.SDOcluststream64
        }[p['float_type']]
        
        self.model = cpp_obj(
            p['k'], p['T'], p['qv'], p['x'], p['chi_min'], p['chi_prop'], p['zeta'],
            p['e'], distance_function, p['seed']
        )
        
        self.last_time = 0
        self.dimension = -1

    def fit_predict(self, X, times=None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Labels for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        labels = np.empty(X.shape[0], dtype=np.int32)
        if self.params['return_sampling']:
            sampling = np.empty(X.shape[0], dtype=np.int32)
            self.model.fit_predict_with_sampling(X, labels, times, sampling)
            return labels, sampling
        else:
            self.model.fit_predict(X, labels, times)
            return labels

    def observer_count(self):
        """Return the current number of observers."""
        return self.model.observer_count()
        
    def get_observers(self, time=None):
        """
        Return observer data.
        
        Returns    
        -------
        data: ndarray, shape (n_observers, n_features)
            Sample used as observer.
            
        observations: ndarray, shape (n_observers,)
            Exponential moving average of observations.
            
        av_observations: ndarray, shape (n_observers,)
            Exponential moving average of observations
            normalized according to the theoretical maximum.

        labels: ndarray, shape (n_observers,)
            Labels / colors of observer.
        """
        if time is None:
            time = self.last_time
        observer_cnt = self.model.observer_count()
        if observer_cnt == 0:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=np.int32), np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type'])
        data = np.empty([observer_cnt, self.dimension], dtype=self.params['float_type'])
        observations = np.empty(observer_cnt, dtype=self.params['float_type'])
        av_observations = np.empty(observer_cnt, dtype=self.params['float_type'])
        labels = np.empty(observer_cnt, dtype=np.int32)
        self.model.get_observers(data, labels, observations, av_observations, self.params['float_type'](time))
        return data, labels, observations, av_observations
