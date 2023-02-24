"""
This python file generates all rule data and stores it in a csv.
"""
import itertools
import numpy as np

name_combinations_items = [("3","4"),("S","D"),("N","Y"),("N","Y"),("A","D")]

def combinations(lst):
    # Generate all possible combinations of items in the tuples
    combos = [''.join(i) for i in list(itertools.product(*lst))]
    return combos

set_combs = combinations(name_combinations_items)
set_types = combinations(name_combinations_items[:2])
set_criteria = combinations(name_combinations_items[2:4])
print(set_criteria)

def possible_strs(set_type):
    possible_values = [str(i) for i in range(1,1+int(set_type[0]))]
    str_length = int(set_type[0]) if set_type[1] == "S" else 2*int(set_type[0])-1
    all_ints = [int(''.join(i)) for i in itertools.product(possible_values, repeat=str_length)]
    return all_ints



""" names = ['Player Name', 'Foo', 'Bar', -1, -1]
scores = ['Score', 250, 500, 100, 30]
np.savetxt('..\data\scores.csv', [p for p in zip(names, scores)], delimiter=',', fmt='%s')
print([x for x in zip(names, scores)]) """