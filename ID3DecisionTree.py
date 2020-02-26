#Using as few loop as possible
import pandas as pd
import numpy as np
from scipy import stats

def entropy(x):
    val, counts = np.unique(x, return_counts=True)
    p = counts/sum(counts)
    return sum(-p*np.log(p))
    
def cond_entropy(x, z, y):
    z0, z1 = y[x <= z], y[x > z]
    pz0 = len(z0)/len(x)
    return pz0*entropy(z0)+(1-pz0)*entropy(z1)

def IG(x, z, y):
    return - cond_entropy(x, z, y)

def vals_and_feas(fea, x, y):
    slices = x[:,fea].copy()
    slices += np.where(slices!=max(slices), 0.5, -0.5)
    return np.fromiter(map(lambda n: IG(x[:,fea], n, y), slices), dtype=float)

def pickin_largest_ig(dataset, y):
    igs = np.array(list(map(lambda x:vals_and_feas(x, dataset, y), range(dataset.shape[1]))))
    return igs, np.unravel_index(igs.argmax(), igs.shape)

def prune(tree, vali_x, vali_y, nround):
    origin_err = (tree.predict(vali_x)!=vali_y).sum()/len(vali_y)
    tree_cp = copy.deepcopy(tree)
    nodes = [tree_cp.root]
    rounds = 0
    while len(nodes)!=0:
        if(rounds==nround):
            break
        rounds += 1
        #print(rounds, nround)
        to_prune = nodes.pop(0)
        left = to_prune.left
        right = to_prune.right
        to_prune.left=None
        to_prune.right=None
        pruned_err = (tree_cp.predict(vali_x)!=vali_y).sum()/len(vali_y)
        if(pruned_err <= origin_err):
            continue
        else:
            to_prune.left = left
            to_prune.right = right 
            nodes.append(to_prune.left)
            nodes.append(to_prune.right)
    return tree_cp

class DecisionTree():
    class DecisionTreeNode():
        def __init__(self, dataset_x, dataset_y, name, left=None, right=None):
            self.x = dataset_x
            self.y = dataset_y
            self.left = left
            self.right =right
            self.decision_val = None
            self.name = name

        def is_pure(self):
            return len(np.unique(self.y))==1

        def is_leaf(self):
            return self.left == None and self.right==None

        def splitting_single(self):
            if self.is_pure():
                return False
            igs, idxs = pickin_largest_ig(self.x, self.y)
            slices = np.where(self.x==self.x.max(axis=0), -0.5, 0.5)
            slices += self.x
            left_idx = self.x[:, idxs[0]] <= slices[idxs[1], idxs[0]]
            right_idx = self.x[:, idxs[0]] > slices[idxs[1], idxs[0]]
            self.decision_val = (idxs[0], slices[idxs[1], idxs[0]])
            self.left = DecisionTree.DecisionTreeNode(self.x[left_idx], self.y[left_idx], self.name+' left')
            self.right = DecisionTree.DecisionTreeNode(self.x[right_idx], self.y[right_idx], self.name+' right')
            return True
        
        def predict_single(self, fea):
            curr = self
            while not curr.is_leaf():
                dec_idx = curr.decision_val[0]
                dec_val = curr.decision_val[1]
                if(fea[dec_idx]>dec_val):
                    curr = curr.right
                else:
                    curr = curr.left
            return stats.mode(curr.y)[0][0]
        
        def predict(self, test):
            return np.array(list(map(self.predict_single, test)))
        
        def __repr__(self):
            return self.name
        
        
    def __init__(self, root):
        self.root = root
        
    def fit(self):
        l = [self.root]
        while(len(l)!=0):
            to_split = l.pop()
            if(to_split.splitting_single()):
                l.append(to_split.left)
                l.append(to_split.right)
            #print(to_split)
        return self
    
    def predict_single(self, fea):
        return self.root.predict_single(fea)
    
    def predict(self, test):
        return self.root.predict(test)
