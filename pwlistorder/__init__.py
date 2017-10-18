import numpy as np
from collections import defaultdict
from itertools import permutations
from copy import deepcopy
import random, cvxpy as cvx

############# utils ##################

def agg_preferences(prefs_list):
    """
    Aggregate a list of pairwise preferences
    
    Take a list of pairwise preferences and aggregate them into a dictionary.
    This is the data structure that the other functions in this package use.
    
    Note
    ----
    The item IDs passed to this function don't strictly need to be integers starting at zero,
    but algorithms that don't take an initial ordering will require that they be when aggregated.
    IDs need to be comparable to each other (e.g. have a method to establish ID1<ID2).
    
    Parameters
    ----------
    prefs_list : list of tuples
        List containing preferences between to items as tuples (ItemId1, ItemId2),
        With ItemId1 being preferred to ItemId2.
        
    Returns
    -------
    defaultdict
        Dictionary with keys as (ItemId1, ItemId2) with ItemId1<ItemId2,
        and values counting the number of times ItemId1 was preferred over ItemId2,
        minus the number of times ItemId2 was preffered over ItemId1.
    """
    aggregated_preferences=defaultdict(lambda: 0)
    for els in prefs_list:
        if els[0]<els[1]:
            aggregated_preferences[(els[0],els[1])]+=1
        else:
            aggregated_preferences[(els[1],els[0])]-=1
    return aggregated_preferences

def eval_ordering(ordering, prefs):
    """
    Score an ordering
    
    Evaluate the number of satisfied preferences minus the number of violated preferences
    under the rank implied by an ordering.
    
    Parameters
    ----------
    ordering : list
        Ordering to be evaluated, with the items in ranked order.
    prefs : defaultdict
        Aggregated preferences (see function 'agg_preferences')
    
    Returns
    -------
    int
        Number of satisfied preferences minus violated preferences of the ordering
    """
    score=0
    cnt=len(ordering)
    for i in range(cnt-1):
        for j in range(i+1,cnt):
            e1,e2=ordering[i],ordering[j]
            if e1<e2:
                score+=prefs[(e1,e2)]
            else:
                score-=prefs[(e2,e1)]
    return score

############# helpers ##################

#Generate all the possible pairs, will be used with minconflict
def _get_pairs_indices(n):
    for i in range(n-1):
        for j in range(i+1,n):
            yield i,j

#Give a relative order between 2 items and see how many constrains will it satisy minus how many will it violate           
def _score_pref(e1_name,e2_name,prefs_dict):
    global e1_name1,e2_name1,prefs_dict1
    e1_name1=e1_name
    e2_name1=e2_name
    prefs_dict1=prefs_dict
    if e1_name<e2_name:
        return prefs_dict[e1_name,e2_name]
    else:
        return -prefs_dict[e2_name,e1_name]
    
#Swap a pair of items in a list
def _swap(lst,pair_tuple):
    lst[pair_tuple[0]],lst[pair_tuple[1]]=lst[pair_tuple[1]],lst[pair_tuple[0]]
    
#Examine the net effect of having a pair as it is vs. the other way around
def _pair_net_effect(lst,pair_ind,prefs_dict):
    lst2=deepcopy(lst)
    e1_ind,e2_ind=pair_ind
    if e1_ind>e2_ind:
        e1_ind,e2_ind=e2_ind,e1_ind
    lst2[e1_ind],lst2[e2_ind]=lst2[e2_ind],lst2[e1_ind]
    score=0
    rev_score=0
    for p1 in range(e1_ind):
        score+=_score_pref(lst[p1],lst[e1_ind],prefs_dict)
        rev_score+=_score_pref(lst2[p1],lst2[e1_ind],prefs_dict)        
        score+=_score_pref(lst[p1],lst[e2_ind],prefs_dict)
        rev_score+=_score_pref(lst2[p1],lst2[e2_ind],prefs_dict)
    for p2 in range(e1_ind+1,e2_ind):
        score+=_score_pref(lst[e1_ind],lst[p2],prefs_dict)
        rev_score+=_score_pref(lst2[e1_ind],lst2[p2],prefs_dict)
        score+=_score_pref(lst[p2],lst[e2_ind],prefs_dict)
        rev_score+=_score_pref(lst2[p2],lst2[e2_ind],prefs_dict)
    for p3 in range(e2_ind+1,len(lst)):
        score+=_score_pref(lst[e1_ind],lst[p3],prefs_dict)
        rev_score+=_score_pref(lst2[e1_ind],lst2[p3],prefs_dict)
        score+=_score_pref(lst[e2_ind],lst[p3],prefs_dict)
        rev_score+=_score_pref(lst2[e2_ind],lst2[p3],prefs_dict)        
    score+=_score_pref(lst[e1_ind],lst[e2_ind],prefs_dict)
    rev_score+=_score_pref(lst2[e1_ind],lst2[e2_ind],prefs_dict)
    return (score,rev_score)
    
############# algorithms ##################

def greedy_order(dict_prefs, list_els):
    """
    Greedy-order
    
    Sort the items according to how many times each item is preferred over any other items.
    
    Note
    ----
    This is implemented here to serve as a handy comparison point, but this heuristic is very simple
    and you can make a much faster implementation with a different data structure than the dict of preferences.
    The time complexity of this implementation is O(items^2).
    
    Parameters
    ----------
    list_els : list
        Items to be ordered
        (e.g. list_els = [i for i in range(nitems)],
        assuming they are enumerated by integers starting at zero)
    dict_prefs : defaultdict
        Aggregated preferences (see function 'agg_preferences')
    
    Returns
    -------
    list
        Ordered list according to this heuristic
    """
    ordering=list()
    els=deepcopy(list_els)
    while els!=[]:
        best_score=float("-infinity")
        for e1 in els:
            score_el=0
            for e2 in els:
                if e1==e2:
                    continue
                score_el+=_score_pref(e1,e2,dict_prefs)
            if score_el>best_score:
                best_score=score_el
                best_el=e1
        ordering.append(best_el)
        els.remove(best_el)
    return ordering

def _kwiksort(list_els, dict_prefs):
    if list_els==[]:
        return []
    pivot=np.random.choice(list_els)
    left=[]
    right=[]
    for el in list_els:
        if el==pivot:
            continue
        else:
            if _score_pref(el,pivot,dict_prefs)<0:
                right.append(el)
            else:
                left.append(el)
    left=_kwiksort(left,dict_prefs)
    right=_kwiksort(right,dict_prefs)
    return left+[pivot]+right

def kwiksort(dict_prefs, list_els, runs=10, random_seed=None):
    """
    Kwik-Sort algorithm
    
    Sort the items with a similar logic as Quick-Sort.
    As there is randomization in the choice of pivots,
    the algorithm is run for multiple times and the best result is returned.
    Time complexity is O(runs * items * log(items)).
    
    Parameters
    ----------
    list_els : list
        Items to be ordered
        (e.g. list_els = [i for i in range(nitems)],
        assuming they are enumerated by integers starting at zero)
    dict_prefs : defaultdict
        Aggregated preferences (see function 'agg_preferences')
    runs : int
        Number of times to run the algorithm
    random_seed : int
        Initial random seed to use
    
    Returns
    -------
    list
        Ordered list according to this heuristic
    """
    best_score=float("-infinity")
    if random_seed is not None:
        np.random.seed(random_seed)
    for run in range(runs):
        ordering=_kwiksort(list_els,dict_prefs)
        score=eval_ordering(ordering,dict_prefs)
        if score>best_score:
            best_score=score
            best_order=ordering
    return best_order

def pagerank(dict_prefs, nitems, eps_search=20):
    """
    PageRank applied to pairwise preferences
    
    The PageRank algorithm is applied by constructing a transition matrix from preferences,
    in such a way that items that are preferred over a given item have 'links' to them.
    Then some small regularization is applied (small probability of any other item being preferred over each item),
    and the regularization that provides the best ordering is returned.
    
    Note
    ----
    This is a naive implementation with dense matrices, initially filled by iterating over the preferences dict,
    and it's not meant for web-scale applications.
    If the number of items is very large (e.g. >= 10^4), you'd be better off implementing this
    with more suitable data structures.
    
    Parameters
    ----------
    nitems : int
        Number of items to be ordered
    dict_prefs : defaultdict
        Aggregated preferences (see function 'agg_preferences').
        Elements must be enumarated as integers starting at zero
    eps_search : int
        Length of search grid for epsilon parameter in (0, 0.5]
    
    Returns
    -------
    list
        Ordered list according to this heuristic
    """
    prefs_mat=np.zeros((nitems,nitems))
    for k,v in dict_prefs.items():
        if v==0:
            continue
        elif v>0:
            prefs_mat[k[1],k[0]]+=v
        else:
            prefs_mat[k[0],k[1]]-=v
    prefs_mat_orig=prefs_mat.copy()
    eps_grid=list(.5**np.logspace(0,1,eps_search))
    best=-10^5
    best_order=None
    
    for eps in eps_grid:
        prefs_mat=prefs_mat_orig.copy()
        for i in range(nitems):
            prefs_mat[:,i]+=eps
            tot=np.sum(prefs_mat[:,i])
            prefs_mat[:,i]=prefs_mat[:,i]/tot

        
        pr=np.ones((nitems,1))/nitems
        for i in range(30):
            pr=prefs_mat.dot(pr)
        lst_pagerank=list(np.argsort(pr.reshape(-1)))
        score_this_order=eval_ordering(lst_pagerank,dict_prefs)
        if score_this_order>best:
            best=score_this_order
            best_order=deepcopy(lst_pagerank)
    return best_order

def cvx_relaxation(dict_prefs, nitems):
    """
    Linear relaxation of the optimization problem
    
    Models the problem as assigning a score to each item, with a hinge loss with arbitrary margin such that,
    for each preference between two items:
    Loss(ItemPref,ItemNonPref)=pos(#prefs * (Score_ItemPref - Score_ItemNonPref) + 1)
    
    where pos(.) is the positive part of a number (x*(x>0))
    
    Note
    ----
    The problem is modeled using cvxpy and solved with its default SDP solver in your computer
    (This is most likely to be ECOS or SC)
    
    Parameters
    ----------
    nitems : int
        Number of items to be ordered
    dict_prefs : defaultdict
        Aggregated preferences (see function 'agg_preferences').
        Elements must be enumarated as integers starting at zero
    
    Returns
    -------
    list
        Ordered list according to this heuristic
    """
    r=cvx.Variable(nitems)    
    obj=0
    for k,v in dict_prefs.items():
        if v>0:
            obj+=cvx.pos(v*r[k[0]]-v*r[k[1]]+1)
        else:
            obj+=cvx.pos(-v*r[k[1]]+v*r[k[0]]+1)
    prob=cvx.Problem(cvx.Minimize(obj))
    prob.solve()
    return list(np.argsort(np.array(r.value).reshape(-1)))

def minconflict(dict_prefs, initial_guess):
    """
    Local search with min-conflict metaheuristic
    
    At each iteration, swaps the pair of items that bring the highest score improvement
    (number of satisfied preferences minus violated preferences).
    Stops when no further improvement is possible.
    
    Time complexity is O(iterations * items^3), thus not suitable for large problems.
    
    Parameters
    ----------
    initial_guess : list
        Initial ordering of the items.
        If you didn't have any ranking criteria or prior ordering beforehand,
        you can get a starting ordering by just shuffling the list of items to be ordered,
        or even use the order induced by some other algorithm from here (e.g. greedy-order)
    dict_prefs : defaultdict
        Aggregated preferences (see function 'agg_preferences').
        Elements must be enumarated as integers starting at zero
    
    Returns
    -------
    list
        Ordered list according to this heuristic
    """
    ordering=deepcopy(initial_guess)
    while True:
        best=0
        best_swap=None
        pairs_indices=_get_pairs_indices(len(ordering))
        for pair_ind in pairs_indices:
            score_as_is,score_rev=_pair_net_effect(ordering,pair_ind,dict_prefs)
            improvement=(score_rev-score_as_is)
            if improvement>best:
                best=improvement
                best_swap=pair_ind
        if best_swap is None:
            break
        else:
            _swap(ordering,best_swap)
    return ordering

def random_swaps(dict_prefs, initial_guess, iterations=50000, repetitions=3, random_seed=None):
    """
    Local search by random swaps
    
    At each iteration, takes two items and swaps them if that improves the ordering.
    Time complexity is O(repetitions * iterations * items + repetitions * items^2)
    
    Parameters
    ----------
    initial_guess : list
        Initial ordering of the items.
        If you didn't have any ranking criteria or prior ordering beforehand,
        you can get a starting ordering by just shuffling the list of items to be ordered,
        or even use the order induced by some other algorithm from here (e.g. greedy-order)
    dict_prefs : defaultdict
        Aggregated preferences (see function 'agg_preferences').
        Elements must be enumarated as integers starting at zero
    iterations : int
        Number of iterations under each run.
        This is the total number of trials regardless of whether the random pair ends up being swapped
    repetitions : int
        Number of times to repeat the procedure.
        Orderings are evaluated after the number of iterations is over, and the best one is returned
    random_seed : int
        Initial random seed to use
    
    Returns
    -------
    list
        Ordered list according to this heuristic
    """
    best=0
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    for rep in range(repetitions):
        ordering=deepcopy(initial_guess)
        for it in range(iterations):
            candidates_ind=random.sample(range(len(ordering)), 2)
            score_as_is,score_rev=_pair_net_effect(ordering,candidates_ind,dict_prefs)
            if score_rev>score_as_is:
                _swap(ordering,candidates_ind)
        score=eval_ordering(ordering,dict_prefs)
        if score>best:
            best=score
            best_ordering=deepcopy(ordering)
    return best_ordering

def metropolis_hastings(dict_prefs, initial_guess, iterations=50000, explore_fact=2, random_seed=None):
    """
    Local search with Metropolis-Hasting metaheuristic
    
    At each iterations, choose a pair of items at random.
    If swapping them improves the ordering, it swaps them.
    If not, it swaps them with a probability inversely proportional to the score decrease of swapping the items.
    Time complexity is O(iterations * items)
    
    Parameters
    ----------
    initial_guess : list
        Initial ordering of the items.
        If you didn't have any ranking criteria or prior ordering beforehand,
        you can get a starting ordering by just shuffling the list of items to be ordered,
        or even use the order induced by some other algorithm from here (e.g. greedy-order)
    dict_prefs : defaultdict
        Aggregated preferences (see function 'agg_preferences').
        Elements must be enumarated as integers starting at zero
    iterations : int
        Number of iterations under each run.
        This is the total number of trials regardless of whether the random pair ends up being swapped
    explore_fact : float
        Parameter for acceptance probability.
        Pairs that don't improve the ordering are swapped with probability:
            p(swap) = explore_fact^score_decrease
    random_seed : int
        Initial random seed to use
    
    Returns
    -------
    list
        Ordered list according to this heuristic
    """
    ordering=deepcopy(initial_guess)
    best=0
    current_score=0
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    for it in range(iterations):
        candidates_ind=random.sample(range(len(ordering)), 2)
        score_as_is,score_rev=_pair_net_effect(ordering,candidates_ind,dict_prefs)
        if score_rev>score_as_is:
            _swap(ordering,candidates_ind)
            current_score+=(score_rev-score_as_is)
            if current_score>best:
                best=current_score
                best_ordering=deepcopy(ordering)
        else:
            criterion=(explore_fact)**(score_rev-score_as_is)
            if np.random.random()<=criterion:
                _swap(ordering,candidates_ind)
                current_score+=(score_rev-score_as_is)
    return best_ordering
