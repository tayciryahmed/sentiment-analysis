# Inspired from https://github.com/bogatyy/cs224d/blob/master/assignment3/tree.py
import math

class Node:  # a node in the tree
    def __init__(self, label = None, word=None):
        self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if leaf
        self.isLeaf = False

class Tree:
    def __init__(self, tokens,label):
        self.root = self.parse(tokens,label)

    def parse(self, tokens, label = None, parent = None):
        # New node
        node = Node(label)
        node.parent = parent
        
        sent_size = len(tokens)

        # leaf Node
        if sent_size == 1:
            node.word = ''.join(tokens[0])
            node.isLeaf = True
            return node

        split = math.floor(sent_size/2)
        node.left = self.parse(tokens[0:split], parent=node)
        node.right = self.parse(tokens[split:], parent=node)

        return node

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words

def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)

def loadTrees(sentences,labels):
    trees = [Tree(sentences[i],labels[i]) for i in range(sentences.shape[0])]
    return trees