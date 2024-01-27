// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_CLUSTERING_WRAPPER_H
#define DSALMON_CLUSTERING_WRAPPER_H

#include "SDOcluststream.h"
// #include "histogram.h"

#include "array_types.h"
#include "distance_wrappers.h"

template<typename FloatType>
class SDOcluststream_wrapper {
    int dimension;
    std::size_t freq_bins;
    SDOcluststream<FloatType> sdoclust; // Use SDOcluststream

public:
    SDOcluststream_wrapper(
      int observer_cnt, 
      FloatType T, 
      FloatType idle_observers, 
      int neighbour_cnt, 
      int chi_min, 
      FloatType chi_prop,
      FloatType zeta, 
      int e,        
      Distance_wrapper<FloatType>* distance, 
      int seed);

    void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
    void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<int>  labels, const NumpyArray1<FloatType> times);
    void fit_predict_with_sampling(const NumpyArray2<FloatType> data, NumpyArray1<int> labels, const NumpyArray1<FloatType> times, NumpyArray1<int> sampled);
    int observer_count();
    void get_observers(NumpyArray2<FloatType> data, NumpyArray1<int> labels, NumpyArray1<FloatType> observations, NumpyArray1<FloatType> av_observations, FloatType time);
};

// Instantiate the class for different floating-point types
DEFINE_FLOATINSTANTIATIONS(SDOcluststream)

#endif
