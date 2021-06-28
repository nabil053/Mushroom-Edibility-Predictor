class Node:
    def __init__(self, depth):
        self.class_of = {}
        self.is_leaf = True
        self.depth = depth
        self.split_on = ''
