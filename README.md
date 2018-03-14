# Ordered lists from pairwise preferences

This is a python package with implementations of algorithms for ordering/ranking a list of items based on aggregated pairwise preferences, in order to solve problems such as “given a set of different Top-N lists of musical artists from different people, construct an overall ranking of musical artists” (note that any ranked list can be decomposed into pairwise preferences).

The idea is to look for a global ordering of the items that would satisfy as many pairwise preferences as possible while violating as few as possible. Orderings are thus evaluated by a score consisting on the number of satisfied preferences minus the number of violated preferences. A brute-force search will always find the best possible ordering (need not be unique) when the number of elements is small (e.g. candidates for a political role with multi-candidate voting schemes), but it becomes intractable with 10 or more elements, in which case other algorithms can produce approximate solutions for this NP-complete problem.

Whenever ordering elements based on such criteria, always keep in mind [Condorcet’s paradox](https://en.wikipedia.org/wiki/Condorcet_paradox) and [Arrow’s impossibility theorem](https://en.wikipedia.org/wiki/Arrow's_impossibility_theorem) when you think about the resulting ordering output by algorithms.

Algorithms here are based solely on preference relationships between items and not on item features. If you are looking for feature-based ranking algorithms (“learning-to-rank” with pairwise methods), you might want to look at other techniques such as RankSVM.

## Algorithms

The following algorithms are implemented:

* Kwik-Sort (`kwiksort`, see [Aggregating inconsistent information: ranking and clustering](https://pdfs.semanticscholar.org/2a25/80233a5e23ca06dcd96fa1e037d014848360.pdf))
* PageRank (`pagerank`, see [Exploiting User Preference for Online Learning in Web Content Optimization Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Bian14Exploiting.pdf))
* Relaxation as a linear problem with hinge loss (`cvx_relax`, see [Convex optimization](https://www.amazon.com/Convex-Optimization-Stephen-Boyd/dp/0521833787))
* Greedy-order (simplest possible heuristic) (`greedy_order`, see [Learning to order things](http://papers.nips.cc/paper/1431-learning-to-order-things.pdf))
* Local search swapping 2 items, guided by:
	* Random choices if they end up improving it (`random_swaps`)
	* Min-conflict (`minconflict`)
	* Metropolis-Hastings (`metropolis_hastings`)

Note that these algorithms might produce very different lists, and their running times and scalability are also very different. If your data is small (e.g. less than 50 items), you might want try local search with min-conflict or metropolis-hastings metaheuristics, while for medium-sizes (e.g. 200 items) min-conflict is already not scalable enough and you might want to stick to metropolis-hastings or try the linear relaxation. For very large lists, only kwik-sort, pagerank and greedy-order will provide good running times.

All the implementation is in Python, so running times are not stellar.

## Installation

Package is available on PyPi, can be installed with

```pip install pwlistorder```

## Usage

The basic data structure used by this package is a dictionary with aggregated preferences between any two items, containing tuples with (ItemId1, ItemId2) as keys, with ItemId1<ItemId2 and values corresponding to the number of times Item1 being preferred over Item2 was observed in the data minus the number of times Item2 being preferred over Item1 was observed in the data (a concatenated list of pairwise preferences coming from different people or different sources). A function to turn lists of preferences into this data structure is also provided (`agg_preferences`), along with a function to score orderings (`eval_ordering`).

For an applied example using simulated data see [this IPython notebook](http://nbviewer.ipython.org/github/david-cortes/pwlistorder/blob/master/example/pwlistorder_example.ipynb)

Example usage
```python
from pwlistorder import agg_preferences, eval_ordering, minconflict, pagerank

# items to order
items = [0, 1, 2]

# fake preferences from different people in the form ‘Item1 is preferred over Item2’
prefs_list = [(0,1), (0,1), (1,0), (0,1), (0,2), (1,2), (2,1), (2,1), (2,0)]

# creating the dictionary of aggregated preferences
dct_prefs = agg_preferences(prefs_list)

# random initial order
starting_order = [1,0,2]

# running min-conflict, requires a starting point
ordering_minconflict = minconflict(dct_prefs, starting_order)

# running pagerank, no starting point required
ordering_pagerank = pagerank(dct_prefs, len(items), eps_search=20)

# comparing results
print(eval_ordering(ordering_minconflict, dct_prefs))
print(eval_ordering(ordering_pagerank, dct_prefs))

# taking a look at the results - be aware of condorcet’s paradox!
print(ordering_minconflict)
print(ordering_pagerank)
```
All functions are documented internally through docstrings (e.g. you can try `help(pagerank)` to see which parameters it takes). The package has only been tested under Python 3.

## References
* Ailon, N., Charikar, M., & Newman, A. (2008). Aggregating inconsistent information: ranking and clustering. Journal of the ACM (JACM), 55(5), 23.
* Bian, J., Long, B., Li, L., Moon, T., Dong, A., & Chang, Y. (2014). Exploiting User Preference for Online Learning in Web Content Optimization Systems. ACM Transactions on Intelligent Systems and Technology (TIST), 5(2), 33.
* Boyd, S., & Vandenberghe, L. (2004). Convex optimization. Cambridge university press.
* Cohen, W. W., Schapire, R. E., & Singer, Y. (1998). Learning to order things. In Advances in Neural Information Processing Systems (pp. 451-457).
