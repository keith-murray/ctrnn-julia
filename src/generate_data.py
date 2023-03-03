"""
This python file generates all rule data and stores it in a csv.
"""
import itertools
import numpy as np

name_combinations_items = [("3","4"),("S","D"),("N","Y"),("N","Y"),("A","D")]
container_dict = {(0,0):["NNA","NYD","YND","YYD"],
                    (0,1):["NND","NYA","YND","YYA"],
                    (1,0):["NND","NYD","YNA","YYA"],
                    (1,1):["NND","NYD","YND","YYA"]}

def combinations(lst):
    # Generate all possible combinations of items in the tuples
    combos = [''.join(i) for i in list(itertools.product(*lst))]
    return combos

def possible_strs(set_type):
    possible_values = [str(i) for i in range(1,1+int(set_type[0]))]
    str_length = int(set_type[0]) if set_type[1] == "S" else 2*int(set_type[0])-1
    all_ints = [''.join(i) for i in itertools.product(possible_values, repeat=str_length)]
    return all_ints

def validate_str(string):
    len_str = len(string)
    set_rep = len(set(string))
    all_same = 1 if set_rep == 1 else 0
    all_diff = 1 if set_rep == len_str else 0
    return all_same, all_diff

def remaining_attributes(string, SET):
    out = string
    for i in SET:
        out = out.replace(i,'',1)
    return out

def validate_left_overs(SET, left_over):
    for i in SET:
        validity = validate_str(left_over+i)
        if sum(validity) != 0:
            return validity
    return validity

def double_set_logic(candidate, left_over):
    keys = candidate.keys()
    result = (0,0)
    for i in keys:
        if sum(candidate[i]) != 0:
            if candidate[i] == left_over[i]:
                result = candidate[i]
            elif sum(candidate[i]) == sum(left_over[i]):
                result = (1,1)
    return result

def validate_double_str(string, set_type):
    candidates = [''.join(i) for i in set([tuple(sorted(i)) for i in itertools.combinations([*string], int(set_type[0]))])]
    candidate_validations = {i:validate_str(i) for i in candidates}
    left_over_validations = {i:validate_left_overs(i, remaining_attributes(string,i)) for i in candidates}
    return double_set_logic(candidate_validations, left_over_validations)

def set_organization(set_type, set_dict, all_strs):
    for i in all_strs:
        if set_type[1] == "D":
            container_mems = [set_type + j for j in container_dict[validate_double_str(i, set_type)]]
        else:
            container_mems = [set_type + j for j in container_dict[validate_str(i)]]
        for k in container_mems:
            set_dict[k].append(int(i))
    return set_dict

def fill_set_dict(set_types, set_dict):
    for i in set_types:
        strs_for_type = possible_strs(i)
        set_dict = set_organization(i, set_dict, strs_for_type)
    return set_dict

def pad_dict(set_dict):
    SETs = []
    max_len = max([len(set_dict[i]) for i in set_dict])
    for i in set_dict:
        current_set = set_dict[i]
        SETs.append([i,]+current_set+[-1 for i in range(max_len-len(current_set))])
    return SETs

def create_SETs():
    set_types = combinations(name_combinations_items[:2])
    set_dict = {i:[] for i in combinations(name_combinations_items)}
    set_dict = fill_set_dict(set_types, set_dict)
    SETs = pad_dict(set_dict)
    np.savetxt('..\data\SETs.csv', [p for p in zip(*SETs)], delimiter=',', fmt='%s')


if __name__ == "__main__":
    create_SETs()