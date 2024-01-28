#ifndef SDOCLUSTSTREAM_H
#define SDOCLUSTSTREAM_H

#include <algorithm>
#include <boost/container/set.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/identity.hpp>
#include <deque>

#include "Vector.h"

template<typename FloatType=double>
class SDOcluststream {
  public:
    typedef std::function<FloatType(const Vector<FloatType>&, const Vector<FloatType>&)> DistanceFunction;

  private:
  
    // number of observers we want
    std::size_t observer_cnt;
    // fraction of observers to consider active
    FloatType active_observers;
    // factor for deciding if a sample should be sampled as observer
    FloatType sampling_prefactor;
    // factor for exponential moving average
    FloatType fading;
    // number of nearest observers to consider
    std::size_t neighbor_cnt;
    // number of nearest observer relative to active_observers
    
    std::vector<FloatType> obs_scaler;

    // counter of processed samples
    int last_index;
    // counter index when we sampled the last time
    int last_added_index;
    // time when we last sampled
    FloatType last_added_time;

    DistanceFunction distance_function;
    std::mt19937 rng;

    std::size_t chi_min;
    FloatType chi_prop;
    FloatType zeta;
    FloatType global_h;
    std::unordered_map<int, FloatType> h;
    std::size_t e; // unused by now

    int last_color;

    struct Observer {
        Vector<FloatType> data;
        FloatType observations;
        FloatType time_touched;        
        FloatType time_added;
        int index;

        FloatType time_cluster_touched;
        int color;

        // Constructor for Observer
        Observer(
            const Vector<FloatType>& _data,
            FloatType _observations,
            FloatType _time_touched,
            FloatType _time_added,
            int _index
            
        ) : data(_data),
            observations(_observations),
            time_touched(_time_touched),
            time_added(_time_added),
            index(_index),
            time_cluster_touched(_time_touched),
            color(0) {}
    };

    struct ObserverCompare{
        FloatType fading;

        // ObserverCompare() : fading(1.0) {}
        ObserverCompare(FloatType fading) : fading(fading) {}

        bool operator()(const Observer& a, const Observer& b) const {
            FloatType common_touched = std::max(a.time_touched, b.time_touched);
            
            FloatType observations_a = a.observations
                * std::pow(fading, common_touched - a.time_touched);
            
            FloatType observations_b = b.observations
                * std::pow(fading, common_touched - b.time_touched);
            
            // tie breaker for reproducibility
            if (observations_a == observations_b)
                return a.index < b.index;
            return observations_a > observations_b;
        }
    } observer_compare;
    
    struct ObserverAvCompare{
        FloatType fading;
        ObserverAvCompare(FloatType fading) : fading(fading) {}
        bool operator()(FloatType now, const Observer& a, const Observer& b) {
            FloatType common_touched = std::max(a.time_touched, b.time_touched);
            
            FloatType observations_a = a.observations * std::pow(fading, common_touched - a.time_touched);
            FloatType age_a = 1-std::pow(fading, now-a.time_added);
            
            FloatType observations_b = b.observations * std::pow(fading, common_touched - b.time_touched);
            FloatType age_b = 1-std::pow(fading, now-b.time_added);
            
            // do not necessarily need a tie breaker here
            return observations_a * age_b > observations_b * age_a;
        }
    } observer_av_compare;

    typedef boost::container::multiset<Observer,ObserverCompare> MapType;
    typedef typename MapType::iterator MapIterator;
    MapType observers;

    // Structure that represents a Cluster (Container of IndexIteratorPairs)
    struct IndexIteratorPair {
        MapIterator it;
        int index; // Derived from it->index

        IndexIteratorPair() {} // Default constructor

        IndexIteratorPair(int index) : index(index) {} 

        IndexIteratorPair(MapIterator it) : it(it) {
            index = it->index;
        }
    };

    struct IndexIteratorHash {
        std::size_t operator()(const IndexIteratorPair& p) const {
            return std::hash<int>{}(p.index);
        }
    };

    struct IndexIteratorEqual {
        bool operator()(const IndexIteratorPair& p1, const IndexIteratorPair& p2) const {
            return p1.index == p2.index;
        }
    };

    typedef std::unordered_set<IndexIteratorPair,IndexIteratorHash,IndexIteratorEqual> IteratorMapType; // represents Iterators of Cluster Observers
    
    typedef std::unordered_map<int, std::pair<IteratorMapType,FloatType>> ClusterMapType; // color, <Cluster, time_added>
    ClusterMapType clusters; 
    
    // Following Structure counts colors of Observers of a Cluster and maintains it ordered (descending)
    struct ColorCount {
        int color;
        int count;

        ColorCount(int color) : color(color), count(1) {}
    };

    struct ColorCountCompare{
        bool operator()(const ColorCount& a, const ColorCount& b) const {            
            if (a.count == b.count) {
                return a.color < b.color;
            }            
            return a.count > b.count;
        }
    };

    typedef boost::multi_index::multi_index_container<
        ColorCount,
        boost::multi_index::indexed_by<
            boost::multi_index::hashed_unique<
                boost::multi_index::member<ColorCount, int, &ColorCount::color>
            >,            
            boost::multi_index::ordered_unique<
                boost::multi_index::identity<ColorCount>,
                ColorCountCompare
            >
        >
    > ColorCountMapType;  

    // Structures for Distance Matrix
    struct IndexDistancePair {
        MapIterator it;
        int index; // Derived from it->index
        FloatType distance;

        IndexDistancePair() {} // Default constructor

        IndexDistancePair(MapIterator it, FloatType distance) : it(it), distance(distance) {
            index = it->index;
        }
    };

    struct DistanceCompare{
        bool operator()(const IndexDistancePair& a, const IndexDistancePair& b) const {
            if (a.distance == b.distance) {
                return a.index < b.index;
            }
            return a.distance < b.distance;
        }
    } distance_compare;

    typedef boost::multi_index::multi_index_container<
        IndexDistancePair,
        boost::multi_index::indexed_by<
            boost::multi_index::hashed_unique<
                boost::multi_index::member<IndexDistancePair, int, &IndexDistancePair::index>
            >,
            // boost::multi_index::ordered_unique<
            //     DistanceCompare
            // >
            boost::multi_index::ordered_unique<
                boost::multi_index::identity<IndexDistancePair>,
                DistanceCompare
            >
        >
    > DistanceMapType;  
    
    typedef std::unordered_map<int, DistanceMapType> DistanceMatrix;
    DistanceMatrix distance_matrix;

    IteratorMapType intersection(const IteratorMapType& set1, const IteratorMapType& set2) {
        IteratorMapType result;

        std::set_intersection(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::inserter(result, result.begin()),
            IndexIteratorEqual()
        );

        return result;
    }

    FloatType CalcJaccard(const IteratorMapType& set1, const IteratorMapType& set2, FloatType time_delta) {
        size_t intersectionSize = intersection(set1, set2).size();
        size_t unionSize = std::pow<FloatType>(fading, time_delta) * (set1.size() + set2.size()) - intersectionSize;

        // Calculate and return the Jaccard similarity
        if ((set1.size() + set2.size()) == 0) {
            return 0.0; // Handle the case where unionSize is 0 to avoid division by zero
        } else {
            return static_cast<FloatType>(intersectionSize) / unionSize;
        }
    }

    void DetermineColor(std::deque<std::pair<IteratorMapType, ColorCountMapType>>& sorted_clusters, FloatType now) {
        std::unordered_set<int> colors_taken;
        while (!sorted_clusters.empty()) {
            const auto& pair = sorted_clusters.front(); // Take the first element
            
            int color = 0;
            IteratorMapType cluster = pair.first;
            ColorCountMapType color_count = pair.second;   

            sorted_clusters.pop_front(); // Remove the first element
            
            if (cluster.size() > e) {                            
                for (auto ccit = color_count.template get<1>().begin(); ccit != color_count.template get<1>().end(); ++ccit) {
                    if (colors_taken.count(ccit->color) < 1) {
                        auto pair1 = clusters[ccit->color];
                        IteratorMapType cluster1 = pair1.first;
                        FloatType time_delta = now - pair1.second;

                        FloatType jaccard = CalcJaccard(cluster, cluster1, time_delta);

                        if (jaccard > 0.5f) {
                            color = ccit->color;
                            colors_taken.insert(color);
                            break;
                        } else {
                            // typedef std::unordered_map<int, std::pair<IteratorMapType,FloatType>> ClusterMapType; // color, <Cluster, time_added>
                            bool setColor = true;
                            for (const auto& pair2 : sorted_clusters) {                            
                                const IteratorMapType& cluster2 = pair2.first;
                                if (CalcJaccard(cluster2, cluster1, time_delta) > jaccard) {
                                    setColor = false;
                                    break;
                                }
                            }
                            if (setColor) {
                                color = ccit->color;
                                colors_taken.insert(color);
                                break;
                            }
                        }
                    }
                }
                if (color == 0) {
                    color = ++last_color;
                }
                
                for (const IndexIteratorPair& indexIteratorPair : cluster) {
                    const MapIterator& it2 = indexIteratorPair.it;
                    if (it2->color != color) {
                        it2->color = color;
                    }
                }

                if (color > 0) {
                    clusters[color] = std::make_pair(cluster, now);
                }
            }
            
        }
    }

    void setObsScaler() {
        FloatType prob0 = 1.0f;
        for (int i = neighbor_cnt; i > 0; --i) {
            prob0 *= static_cast<FloatType>(i) / (observer_cnt+1 - i);
        }

        obs_scaler[observer_cnt] = 1.0f;
        FloatType prob = prob0;

        int current_neighbor_cnt = neighbor_cnt;
        
        for (int i = observer_cnt - 1; i > 0; --i) {
            prob *= static_cast<FloatType>(i+1) / static_cast<FloatType>((i+1)-current_neighbor_cnt);

            int current_neighbor_cnt_target = (static_cast<FloatType>(i-1)) / static_cast<FloatType>((observer_cnt-1)) * neighbor_cnt + 1;   
            while (current_neighbor_cnt > current_neighbor_cnt_target) {      
                prob *= static_cast<FloatType>(i+1-current_neighbor_cnt) / static_cast<FloatType>(current_neighbor_cnt);

                current_neighbor_cnt--;
            }
            obs_scaler[i] = prob0 / prob;
        }
        obs_scaler[0] = prob0;
    }

    // Calc Threshold for Graph Edge Cutting
    FloatType UpdateH(int keyIndex, size_t n, const MapIterator& last_active_observer) {
        // Check if the key 'index' exists in the unordered_map
        auto it = distance_matrix.find(keyIndex);
        if (it != distance_matrix.end()) {
            // Access the DistanceMapType associated with the key 'index'
            const DistanceMapType& distanceMap = it->second;

            if (distanceMap.empty()) {  
                return FloatType();
            }

            // Using the by_distance index to find the nth lowest distance
            // auto iterator = distanceToObservers.get<by_distance>();
           
            auto dit = distanceMap.template get<1>().begin();
            // std::advance(dit, n - 1);
            int j = 0;
            for (dit; dit != distanceMap.template get<1>().end(); ++dit) {
                MapIterator it = dit->it;
                if (!observer_compare(*last_active_observer, *it)) {
                    j++;
                }
                if (j == n) {
                    break;
                }
            }

            if (dit == distanceMap.template get<1>().end()) {  
                // dit = distanceMap.template get<1>().end();
                dit--;
                // return rit->distance;                 
            }

            h[keyIndex] = dit->distance;
            
            return dit->distance;   

        } else {
            // Handle the case where key 'i' is not found
            std::cerr << "Error: Key 'i' not found in the unordered_map." << std::endl;
        }

        // Return a default value or handle the error as needed
        return FloatType(); // You can return a default value or an appropriate error handling strategy
    }

    // Maintain Distance Matrix
    bool AddIndexDistancePairToMap(int keyIndex, const IndexDistancePair& pair) {
        // Find the map associated with the given key 'index'
        auto it = distance_matrix.find(keyIndex);
        if (it != distance_matrix.end()) {
            // Add the IndexDistancePair to the found DistanceMapType
            it->second.insert(pair);
        } else {
            // Handle the case where key 'index' is not found
            std::cerr << "Error: Key 'index' not found in the unordered_map." << std::endl;
        }

        return false; // Adding unsuccessful
    }

    // Maintain Distance Matrix
    bool RemoveIndexDistancePairFromMap(int keyIndex, int pairIndex) {
        // Find the map associated with the given key 'index'
        auto it = distance_matrix.find(keyIndex);
        if (it != distance_matrix.end()) {
            // Access the DistanceMapType associated with the key 'index'
            DistanceMapType& distanceMap = it->second;

            // Find the element matching the pair_index and remove it
            auto itToRemove = distanceMap.template get<0>().find(pairIndex);
            if (itToRemove != distanceMap.template get<0>().end()) {
                distanceMap.template get<0>().erase(itToRemove);
                return true; // Removal successful
            } else {
                // Handle the case where pair_index is not found
                std::cerr << "Error: Pair with pair_index not found in the map." << std::endl;
            }
        } else {
            // Handle the case where key 'index' is not found
            std::cerr << "Error: Key 'KeyIndex' not found in the unordered_map." << std::endl;
        }

        return false; // Removal unsuccessful
    }
    
    bool HasEdge(FloatType distance, const MapIterator& it) {
        return  distance < (zeta * h[it->index] + (1 - zeta) * global_h);
    }
    
    void DFS(IteratorMapType& cluster, ColorCountMapType& color_count, const MapIterator& it, const MapIterator& last_active_observer, FloatType now) {
        // Check if the current iterator is not less than or equal to the last_active_observer

        auto cit = cluster.find(IndexIteratorPair(it));
        if ( !observer_compare(*last_active_observer, *it) && (cit == cluster.end()) ) {
            // Add it to the cluster if it is not yet there and it is active
            
            auto ccit = color_count.template get<0>().find(it->color);
            if (ccit != color_count.template get<0>().end()) {
                color_count.template get<0>().modify(ccit, [](ColorCount& cc) { cc.count++; });                
            } else {
                color_count.insert(ColorCount(it->color));
            }

            it->time_cluster_touched = now;            
            cluster.insert(IndexIteratorPair(it));

            auto mit = distance_matrix.find(it->index);

            // Check if the key exists in the map
            if (mit != distance_matrix.end()) {
                // Access the associated 'DistanceMapType'
                const DistanceMapType& distanceMap = mit->second;

                // Now, you can iterate over 'distanceMap'

                for (const auto& dPair : distanceMap.template get<1>()) {
                    FloatType distance = dPair.distance;
                    if (HasEdge(distance, it)) {
                        MapIterator it1 = dPair.it;
                        if (HasEdge(distance, it1)) {
                            DFS(cluster, color_count, it1, last_active_observer, now);
                        }
                    } else {
                        break;
                    }
                }
            } 
        }
    }

    // Main method
    int fitPredict_impl(const Vector<FloatType>& data, FloatType now, bool fit_only) {
        FloatType score = 0; // returned for first seen sample
        std::unordered_map<int, FloatType> label_vector;
        int label (0);
        
        DistanceMapType nearest;
        
        int i = 0;
        int active_threshold = (observers.size()-1) * active_observers; // means active_threshold+1 active observers
        std::size_t chi = std::max(static_cast<std::size_t>(observers.size() * chi_prop), chi_min);
        int current_neighbor_cnt = 
            (observers.size() == observer_cnt) ?
            neighbor_cnt :
            (static_cast<FloatType>(observers.size()-1)) / static_cast<FloatType>((observer_cnt-1)) * neighbor_cnt + 1;

        MapIterator worst_observer = observers.begin();

        MapIterator last_active_observer = observers.begin();
        std::advance(last_active_observer, active_threshold);
        
        FloatType observations_sum = 0;

        global_h = 0;

        // std::unordered_map<int, MapIterator> cluster_header;

        for (auto it = observers.begin(); it != observers.end(); ++it) {
            // std::cout << "(" << it->index << ", " << it->color << ") ";

            // sorts clusters by size, if tied by size of most dominant color of its observers, and if then tied by lower dominant color
            // idea clusters are sorted by being most crisply defined such that label is inherited easily
            auto cmp_cluster = [](const std::pair<IteratorMapType, ColorCountMapType>& a, const std::pair<IteratorMapType, ColorCountMapType>& b) -> bool {
                if (a.first.size() == b.first.size())
                {
                    auto ccit_a = a.second.template get<1>().begin();
                    auto ccit_b = b.second.template get<1>().begin();
                    ColorCount ca = *ccit_a;
                    ColorCount cb = *ccit_b;
                    if (ca.count == cb.count) {
                        return ca.color < cb.color;
                    }            
                    return ca.count > cb.count;
                }
                return a.first.size() > b.first.size();
            };
            std::deque<std::pair<IteratorMapType, ColorCountMapType>> sorted_clusters;

            FloatType distance = distance_function(data, it->data);
            observations_sum += it->observations * std::pow<FloatType>(fading, now-it->time_touched);
            
            nearest.insert(IndexDistancePair(it, distance));

            if (i <= active_threshold) {
                global_h += UpdateH(it->index, chi, last_active_observer) / static_cast<FloatType>(active_threshold+1);
                
                if (i == active_threshold) {
                    // std::cout << std::endl;
                    for (auto it1 = observers.begin(); it1 != observers.end(); ++it1) {
                        if (observer_compare(*last_active_observer, *it1)) {
                            break;
                        }
                        if (it1->time_cluster_touched < now) { 
                            IteratorMapType cluster;
                            ColorCountMapType color_count; // counts the known labels of observers of the cluster                            
                            DFS(cluster, color_count, it1, last_active_observer, now); // find the connected component (cluster)                  
                            sorted_clusters.push_back(std::make_pair(cluster, color_count));
                        }
                    }       

                    std::sort(sorted_clusters.begin(),sorted_clusters.end(), cmp_cluster);
                    
                    DetermineColor(sorted_clusters, now);

                    if (!fit_only) {
                        // set outlier score, count labels
                        int j = 0;
                        auto nit = nearest.template get<1>().begin();
                        while (nit != nearest.template get<1>().end() && j < current_neighbor_cnt) {
                            auto it1 = nit->it;
                            int color = it1->color;

                            if (color > 0) {
                                if (label_vector.find(color) != label_vector.end()) {
                                    label_vector[color] = 0.0f;
                                }
                                label_vector[color] += 1.0f / static_cast<FloatType>(current_neighbor_cnt);
                            }

                            if (j == (current_neighbor_cnt/2)) {
                                if (current_neighbor_cnt % 2 == 0) {
                                    score += 0.5 * nit->distance;;
                                } else {
                                    score += nit->distance;
                                }

                                score += nit->distance;
                            }  
                            if (j == (current_neighbor_cnt/2+1) && current_neighbor_cnt % 2 == 0) {
                                score += 0.5 * nit->distance;
                            }
                                
                            ++j;
                            ++nit;
                        }

                        // set labels
                        FloatType max_value = std::numeric_limits<FloatType>::min();
                        for (auto entry : label_vector) {
                            int color = entry.first;
                            FloatType value = entry.second;

                            if (value > max_value) {
                                max_value = value;
                                label = color;  // Update argmax_color with the color corresponding to the maximum value
                            } else if (value == max_value) {

                                j = 0;
                                nit = nearest.template get<1>().begin();
                                while (nit != nearest.template get<1>().end() && j < current_neighbor_cnt) {
                                    auto it1 = nit->it;

                                    if (it1->color == label) {
                                        break;
                                    }
                                    if (it1->color == color) {
                                        label = color;
                                        break;
                                    }
                                    ++j;
                                    ++nit;
                                }
                            }
                        }
                        // std::cout << std::endl;
                    }
                }
            }

            if (observer_av_compare(now, *worst_observer, *it)) {
                worst_observer = it;
            }
            i++;
        }

        // update Observer observations of nearest Observers
        FloatType observations_nearest_sum = 0;
        int j = 0;
        auto observed = nearest.template get<1>().begin();
        while (observed != nearest.template get<1>().end() && j < current_neighbor_cnt) {
            MapIterator it = observed->it;    

            auto node = observers.extract(it);            
            Observer& observer = node.value();
            observer.observations *= std::pow<FloatType>(fading, now-observer.time_touched);
            //observer.observations += 1;
            observer.observations += obs_scaler[observers.size()];
            observer.time_touched = now;
            observations_nearest_sum += observer.observations;
            observers.insert(std::move(node));
            // take into account that observations have been
            // incremented by 1 for observations_nearest_sum

            // observations_sum += 1;
            observations_sum += obs_scaler[observers.size()];

            ++j;
            ++observed;        
        }
        
        // add new Observer randomly
        bool add_as_observer = 
            observers.empty() ||
            (rng() - rng.min()) * observations_sum * (last_index - last_added_index) < sampling_prefactor * (rng.max() - rng.min()) * observations_nearest_sum * (now - last_added_time) ;

        if (add_as_observer) {
            if (observers.size() < observer_cnt) {                
                worst_observer = observers.insert(Observer(data, obs_scaler[observers.size()], now, now, last_index)); // to add to the distance matrix
            } else {
                
                // remove worst observer from distance matrix (row)
                if (distance_matrix.find(worst_observer->index) != distance_matrix.end()) {
                    distance_matrix.erase(worst_observer->index);
                } else {
                    std::cout << "Key " << worst_observer->index << " not found in the unordered map." << std::endl;
                }

                // remove worst observer from the distance vector of the new observer
                auto it = nearest.template get<0>().find(worst_observer->index);
                if (it != nearest.template get<0>().end()) {
                    nearest.template get<0>().erase(it);
                }

                // remove worst observer from distance matrix (col)
                for (auto nit = nearest.template get<0>().begin(); nit != nearest.template get<0>().end(); ++nit) { 
                    RemoveIndexDistancePairFromMap(nit->index, worst_observer->index);                 
                }
                
                auto node = observers.extract(worst_observer);
                Observer& observer = node.value();
                observer.data = data;
                // observer.observations = 1;
                observer.observations = obs_scaler[observers.size()];
                observer.time_touched = now;
                observer.time_added = now;
                observer.time_touched = now;
                observer.index = last_index;
                observer.color = 0;
                observers.insert(std::move(node));                
            }

            // add new observer to distance matrix (col)
            for (auto nit = nearest.template get<0>().begin(); nit != nearest.template get<0>().end(); ++nit) {                                     
                AddIndexDistancePairToMap(nit->index, IndexDistancePair(worst_observer, nit->distance));                    
            }

            // add new observer to distance matrix (row)
            distance_matrix[last_index] = nearest;

            last_added_index = last_index;
            last_added_time = now;            
        }
        // std::cout << std::endl;
        
        
        // coutCounter++;
        // std::cout << "Counter Value: " << coutCounter << std::endl;
        last_index++;
        // Result result ({score, label_vector, label});
        if (score>0) {
            // do nothing
        }
        if (label>0) {
            // do nothing
        }

        return label;
    }

public:
    SDOcluststream(
        std::size_t observer_cnt, 
        FloatType T, 
        FloatType idle_observers, 
        std::size_t neighbor_cnt,
        std::size_t chi_min,
        FloatType chi_prop,
        FloatType zeta,
        std::size_t e,
        SDOcluststream<FloatType>::DistanceFunction distance_function = Vector<FloatType>::euclidean, 
        int seed = 0
    ) : observer_cnt(observer_cnt), 
        active_observers(1-idle_observers), 
        sampling_prefactor(observer_cnt * observer_cnt / neighbor_cnt / T),
        fading(std::exp(-1/T)),
        neighbor_cnt(neighbor_cnt),
        obs_scaler(observer_cnt+1),
        last_index(0),
        last_added_index(0),
        distance_function(distance_function),
        rng(seed),
        chi_min(chi_min),
        chi_prop(chi_prop),
        zeta(zeta),
        global_h(0),
        h(),
        e(e),
        last_color(0),
        observer_compare(fading),        
        observer_av_compare(fading),
        observers(observer_compare),  // Initialize observers container with initial capacity and comparison function
        clusters(),
        distance_compare(),
        distance_matrix()  
    {
        setObsScaler();
        // std::cout << "obs_scaler elements: " << std::endl;
        // for (const auto& element : obs_scaler) {
        //     std::cout << element << " ";
        // }
        // std::cout << std::endl;
    }

    void fit(const Vector<FloatType>& data, FloatType now) {
        fitPredict_impl(data, now, true);
    }

    FloatType fitPredict(const Vector<FloatType>& data, FloatType now) {
        return fitPredict_impl(data, now, false);
    }
    
    int observerCount() { return observers.size(); }
    
    bool lastWasSampled() { return last_added_index == last_index - 1; }

    class ObserverView{
        FloatType fading;
        MapIterator it;
    public:
        ObserverView(FloatType fading, MapIterator it) :
            fading(fading),
            it(it)
        { }
        // int getIndex() {return it->index};
        Vector<FloatType> getData() { return it->data; }
        int getColor() { return it->color; }
        FloatType getObservations(FloatType now) {
            return it->observations * std::pow(fading, now - it->time_touched);
        }
        FloatType getAvObservations(FloatType now) {
            return (1-fading) * it->observations * std::pow(fading, now - it->time_touched) /
                (1-std::pow(fading, now - it->time_added));
        }
    };

    class iterator : public MapIterator {
        FloatType fading;
      public:
        ObserverView operator*() { return ObserverView(fading, MapIterator(*this)); };
        iterator() {}
        iterator(FloatType fading, MapIterator it) : 
            MapIterator(it),
            fading(fading)
        { }
    };

    iterator begin() { return iterator(fading, observers.begin()); }
    iterator end() { return iterator(fading, observers.end()); }
};                              

#endif  // SDOCLUSTSTREAM_H
