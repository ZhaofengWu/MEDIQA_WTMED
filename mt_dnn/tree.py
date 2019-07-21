from nltk.tree import Tree as NLTKTree

NON_BREAKING_SPACE = u'\xa0'
NON_BREAKING_SPACE_ESCAPE = '<NBS>'

class Tree:
    def __init__(self, content=None, str_repr=None, idx_as_leaf=None):
        self.children = []
        self.content = content
        if self.content is not None:
            self.content = self.content.replace(NON_BREAKING_SPACE_ESCAPE, NON_BREAKING_SPACE)
        self.str_repr = str_repr  # only present when this is root
        self.idx_as_leaf = idx_as_leaf  # only present when this is leaf

    def leaves(self):
        if self.idx_as_leaf is not None:
            return {self}
        else:
            leaves_ = set()
            for child in self.children:
                leaves_ |= child.leaves()
            return leaves_

    def __str__(self):
        return ' '.join([leaf.content for leaf in self.leaves()])

    def __repr__(self):
        return self.__str__()

    def char_indices(self):
        return [ord(char) for char in self.str_repr]

    def assert_binary(self):
        if len(self.children) > 0:
            if len(self.children) != 2:
                # only allow 1 child for 1-element trees
                assert len(self.children) == 1 and self.str_repr is not None and len(self.children[0].children) == 0
            for child in self.children:
                child.assert_binary()

    @classmethod
    def from_char_indices(cls, indices):
        string = ''.join([chr(i) for i in indices])
        return cls.from_string(string)

    @classmethod
    def from_string(cls, string):
        return cls.from_nltk_tree(NLTKTree.fromstring(string), string)

    @classmethod
    def from_nltk_tree(cls, nltk_tree, str_repr):
        tree, num_leaves = cls.from_nltk_tree_helper(nltk_tree, next_leaf_idx=0)
        tree.str_repr = str_repr
        assert len(nltk_tree.leaves()) == num_leaves
        return tree

    @classmethod
    def from_nltk_tree_helper(cls, nltk_tree, next_leaf_idx):
        if not isinstance(nltk_tree, NLTKTree):  # leaf case
            return Tree(content=nltk_tree, idx_as_leaf=next_leaf_idx), next_leaf_idx + 1
        else:
            t = Tree()
            expected_final_idx = next_leaf_idx + len(nltk_tree.leaves())
            for child in nltk_tree:
                child_tree, next_leaf_idx = cls.from_nltk_tree_helper(child, next_leaf_idx)
                t.children.append(child_tree)
            assert next_leaf_idx == expected_final_idx
            assert len(t.children) == len(nltk_tree)
            return t, next_leaf_idx
