class Node:
    def __init__(self, df, split_on=None):
        self.class_of = {}
        self.is_leaf = True
        self.split_on = ''
        