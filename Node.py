class Node:
    def __init__(self, split_on=None):
        self.class_of = {}
        self.is_leaf = True
        self.split_on = split_on
