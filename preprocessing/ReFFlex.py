import numpy as np
import networkx as nx
import torch_geometric
import scipy.sparse as sp
import torch
from tqdm import tqdm
from numba import jit, prange
import time
#np.set_default_dtype(np.float16)

@jit(nopython=True)
def add_edge(edge_index, u, v):
    # Depreciated
	new_edge = np.array([[u], [v]])
	return np.concatenate((edge_index, new_edge), axis=1)

# def dense_adj_rw_normalize_v2(A):
#     # Depreciated
#     A = sp.coo_matrix(A)
#     rowsum = np.array(A.sum(1)) # D
#     d_inv = np.power(rowsum, -1).flatten() # D^-1
#     d_inv[np.isinf(d_inv)] = 0.
#     d_mat_inv = sp.diags(d_inv) # D^-1
#     return A.dot(d_mat_inv).transpose() # D^-1A 
    
# def mapRicci(Adj, W, beta):
#     # Depreciated
#     Adj = Adj.astype(np.float16)
#     W = W.astype(np.float16)
#     n= Adj.shape[0]
#     D = sum(Adj)
#     A0 = Adj.reshape(n, 1, n)
#     mapTri = np.multiply(Adj, A0) # A3
#     print("mapTri calculated\r")
#     # mapsumTri[i, j] -> [0, 0, 1, ..., 0, 0]
#     A2 = np.matmul(Adj, Adj) * (1-np.eye(n)).astype(np.float16)
#     mapnumTri = np.multiply(A2, Adj).astype(np.float16) # A2F
#     print("mapnumTri calculated\r")
#     mapPsumTri = np.multiply(np.matmul(W, Adj).transpose(), Adj).astype(np.float16)
#     print("mappsumTri calculated\r")


#     MindN = np.multiply(D, Adj).astype(np.float16)
#     np.place(MindN, MindN == 0, np.inf)
#     mapMindN = np.min(MindN , axis=1)
#     print("mapMindN calculated\r")

#     mapRec = np.clip(np.multiply(A2, A0).astype(np.float16)-mapTri-1, 0, np.inf).astype(np.float16)
#     print("mapRec calculated\r")

#     A5 = np.matmul(mapRec>0, W.transpose()).astype(np.float16)

#     mapgammaTri = np.zeros((n, n), dtype=np.float16)
#     mapgammaRec = np.zeros((n, n), dtype=np.float16)
#     mapPsumRec = np.zeros((n, n), dtype=np.float16)
#     A4 = np.dot(mapTri, mapnumTri) 
#     for i in tqdm(range(n)):
#         for j in range(n):
#             if Adj[i, j] > 0:
#                 mapgammaTri[i, j] = A4[i, j, i] - mapnumTri[i, j]
#                 mapgammaRec[i, j] = sum(mapRec[j, i])
#                 mapPsumRec[i, j] = A5[j, i, j]
                
#     mapRic = ((mapgammaRec * mapPsumRec + mapgammaTri * mapPsumTri)/mapMindN.reshape(n, 1) + mapPsumTri / beta + (1-W).transpose()/(D + 1)) / (D+1).reshape(n, 1) ** 0.5
#     np.place(mapRic, Adj == 0, np.inf)
#     return mapRic

# def rewiring_v2(edge_index, beta, kappa, edge_type=None):
#     # Depreciated
#     m = edge_index.shape[1]
#     n = np.max(edge_index) + 1
    
#     # transfer to dense matrix
#     if edge_type is None:
#         edge_type = np.zeros(m, dtype=np.int64)
#     Adj = torch_geometric.utils.to_scipy_sparse_matrix(torch.tensor(edge_index))
#     Adj = Adj.todense()
#     Adj = np.array(Adj)
    
#     # generate RW probability model.
#     W = dense_adj_rw_normalize_v2(Adj)
#     W = np.array(W.todense())
    
#     mapCurv = mapRicci(Adj, W, beta=beta)
    
    
#     edge_index, edge_type = skip_connection(Adj, mapCurv, kappa=kappa, edge_index=edge_index, edge_type=edge_type)
#     return edge_index, edge_type





def rw_normalize_v3(A):
    # a random-walk normalization of A
    # A: sparse matrix
    # W: sparse matrix
    rowsum = np.array(A.sum(1)) # D
    d_inv = np.power(rowsum, -1).flatten() # D^-1
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv) # D^-1
    return A.dot(d_mat_inv).transpose() # D^-1A 


def mapRicci_v3(A, W, beta):
    # Get the ricci curvature adjacency matrix
    n = A.shape[0]
    mapPsumTri = W.dot(A).transpose()
    A_d = np.array(A.todense())
    A2 = A.dot(A)
    A2_d = np.array(A2.todense())
    A2_d2 = (A2_d > 1)
    I = np.eye(n)

    W_d = np.array(W.todense())
    Model = PARicci2(A_d, W_d)

    D = sum(A_d)
    MindN = np.multiply(D, A_d)
    np.place(MindN, MindN == 0, np.inf)
    mapMindN = np.min(MindN , axis=1)

    mapgammaTri = np.zeros((n, n))
    mapPsumRec = np.zeros((n, n))
    mapgammaRec = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if A_d[i,j] > 0:
                numTri = np.argwhere(np.multiply(A_d[i], A_d[j])>0)
                if numTri.shape[0] > 0:
                    mapgammaTri[i, j] += sum(A2_d[i, numTri[:, 0]]) - numTri.shape[0]
            
                numRec = np.argwhere(np.multiply((A_d[i]-A_d[j]-I[j]), A2_d2[j])>0)
                if numRec.shape[0] > 0:
                    mapgammaRec[j, i] += sum(A2_d[j, numRec[:, 0]]) - numRec.shape[0]
                    mapPsumRec[j, i] += sum(W_d[i, numRec[:, 0]])
                    
                
    mapRic = ((mapgammaRec * mapPsumRec + np.multiply(mapgammaTri, mapPsumTri.todense()))/mapMindN.reshape(n, 1)+ mapPsumTri.todense() / beta + (1-W_d).transpose()/(D + 1)) / (D+1).reshape(n, 1) ** 0.5
    np.place(mapRic, A_d == 0, -np.inf)
    
    return mapRic

def add_edge_step_v1(adj, ric, edge_index, edge_type):
    I = np.eye(adj.shape[0])
    ric[ric<=0.05]=np.inf
    ind_min = np.argmin(ric)
    ind_x = int(np.floor(ind_min/ric.shape[0]))
    ind_y = ind_min - ind_x*ric.shape[0]
    neighbour = np.argwhere(adj[ind_y] - I[ind_x] > 0)
    if neighbour.shape[0]>0:
        newedge = np.array([[ind_x] * neighbour.shape[0], neighbour[:, 0]])
        edge_index = np.concatenate((edge_index, newedge), axis=1)
        edge_type = np.concatenate((edge_type, np.array([1]*neighbour.shape[0])))
    return edge_index, edge_type

def skip_connection_v3(adj, ric, kappa, edge_index, edge_type):
    I = np.eye(adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
#            if i != j and ric[i, j] != -np.inf and 0.01 < ric[i, j] < kappa:
            if i != j and ric[i, j] != -np.inf and ric[i, j] < kappa:
                #neighbour = np.argwhere(adj[j] - adj[i] - I[i] > 0)
                neighbour = np.argwhere(adj[j] - I[i] > 0)
                if neighbour.shape[0]>0:
                    newedge = np.array([[i] * neighbour.shape[0], neighbour[:, 0]])
                    edge_index = np.concatenate((edge_index, newedge), axis=1)
                    edge_type = np.concatenate((edge_type, np.array([1]*neighbour.shape[0])))
    
    return edge_index, edge_type


def rewiring_v3(edge_index, beta, kappa, edge_type=None):
    # rewire the graph
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    
    if edge_type is None:
        edge_type = np.zeros(m, dtype=np.int64)
    A = torch_geometric.utils.to_scipy_sparse_matrix(torch.tensor(edge_index))
    W = rw_normalize_v3(A)
    timestart = time.time()
    mapCurv = mapRicci_v3(A, W, beta=beta)
    timeend = time.time()
    edge_index, edge_type = skip_connection_v3(np.array(A.todense()), mapCurv, kappa=kappa, edge_index=edge_index, edge_type=edge_type)
    
    return edge_index, edge_type, timeend-timestart

def rewiring_step_v1(edge_index, beta, edge_type=None):
    # rewire the graph by step
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    if edge_type == None:
        edge_type = np.zeros(m, dtype=np.int64)
    A = torch_geometric.utils.to_scipy_sparse_matrix(torch.tensor(edge_index))
    W = rw_normalize_v3(A)
    mapCurv = mapRicci_v3(A, W, beta=beta)
    edge_index, edge_type = add_edge_step_v1(np.array(A.todense()), mapCurv, edge_index=edge_index, edge_type=edge_type)
    
    return edge_index, edge_type


class PARicci2():
    
    def __init__(self, A: np.array, W: np.array, V=None):
        if (len(A.shape) != 2) or (A.shape[0] != A.shape[1]):
            raise ValueError("invalid shape for Adjacency matrix.")
        if np.sum(A - A.transpose()) != 0:
            raise ValueError("invalid data for Adjacency matrix.")
        
        self.num_V = A.shape[0]
        self.I = np.eye(self.num_V)
        self.A = A
        self.W = W
        #self.W_norm = u.prob_normalization(W)
        #self.nxG = nx.Graph(self.W_norm)
        #nx.set_edge_attributes(self.nxG, 0, "hL")
        #self.POS = nx.spring_layout(self.nxG)
        if V != None:
            self.is_V = True
            self.V = V
        else:
            self.is_V = False
        self.D = sum(self.A)
        
    def d(self, i):
        
        return self.D[i]
            
    def Shortest_dist(self):
        Sp = self.A
        mask = Sp
        I = np.eye(self.num_V)
        for i in range(self.num_V - 1):
            m_next = np.matmul(mask, self.A)
            res = ((1 - mask) * m_next > 0) * 1 - I
            Sp = Sp + (i + 2) * res
            mask = mask + res
        return Sp
    
    def S(self, r, i):
        mask = self.A[i]
        I = np.eye(self.num_V)[i]
        res = mask
        for step in range(r - 1):
            m_next = np.matmul(mask, self.A)
            res = ((1 - mask) * m_next > 0) * 1 - I
            mask = mask + res
        return res
    
    def B(self, r, i):
        mask = self.A[i]
        I = np.eye(self.num_V)[i]
        res = mask
        for step in range(r - 1):
            m_next = np.matmul(mask, self.A)
            res = ((1 - mask) * m_next > 0) * 1 - I
            mask = mask + res
        return mask
    
    def is_inS(self, j, r, i):
        if j >= self.num_V or i >= self.num_V:
            raise ValueError("No such node {} or {}".format(i, j))
        
        Svec = self.S(r=r, i=i)
        return Svec[j]!=0
    
    def is_inB(self, j, r, i):
        if j >= self.num_V or i >= self.num_V:
            raise ValueError("No such node {} or {}".format(i, j))
        
        Bvec = self.B(r=r, i=i)
        return Bvec[j]!=0
    
    def num_Triangle(self, i, j):
        if self.A[i, j] != 1:
            raise ValueError("No direct connection between {} and {}".format(i, j))
        
        return self.S(1, i) * self.S(1, j)
    
    def psum_Triangle(self, i, j):

        return np.matmul(self.W[j], self.num_Triangle(i, j))
    
    def gammaTri(self, i, j):
        if self.A[i, j] != 1:
            raise ValueError("No direct connection between {} and {}".format(i, j))
        
        gamma = 0
        for k in np.argwhere(self.num_Triangle(i, j)>0)[:, 0]:
            gamma += sum(self.S(1, k)*self.S(1, i)) - 1
        return gamma
        
    def num_Rectangle(self, i, j):
        #if self.A[i, j] != 1:
        #    raise ValueError("No direct connection between {} and {}".format(i, j))
        
        I = np.eye(self.num_V)
        S1i = self.S(1, i)
        S1j = self.S(1, j)
        kvec = (S1i - I[j] - S1j)
        recvec = np.zeros(self.num_V, dtype=int)
        for k in np.argwhere(kvec > 0)[:, 0]:
            if 1 in ((self.S(1, k) * S1j - I[i]) > 0):
                recvec[k] = 1
        
        return recvec
    
    def psum_Rectangle(self, i, j):
        # num_Rec(i, j) -> #^i (i,j)
        # psum_Rec(i, j) <- #^j(i, j)
        return np.matmul(self.W[j], self.num_Rectangle(j, i))

    def gammaRec(self, i, j):
        if self.A[i, j] != 1:
            raise ValueError("No direct connection between {} and {}".format(i, j))
        
        gamma = 0
        for k in np.argwhere(self.num_Rectangle(j, i)>0)[:, 0]:
            gamma += sum(self.S(1, k)*self.S(1, i)) - 1   
        return gamma 
            
    def mindn(self, i):
        mind = 99
        for k in np.argwhere(self.S(1, i) > 0)[:, 0]:
            if self.d(k) < mind:
                mind = self.d(k)
                
        return mind
    
    def PARic(self, i, j, beta):
        # Ric(i, j) = 1/sq(d_i+1) * ((1-P_ji)/(d_j+1)+psumTri/beta+(gammaRecj*psumRec+gammaTrij*psumTri)/mindn_i)
        d_i = self.D[i]
        d_j = self.D[j]
        psumTri = self.psum_Triangle(i, j)
        gammaTrij = self.gammaTri(i, j)
        psumRec = self.psum_Rectangle(i, j)
        gammaRecj = self.gammaRec(i, j)
        mindni = self.mindn(i)
        
        sep1 = (1 - self.W[j, i]) / (d_j + 1)
        
        sep2 = psumTri / beta
        
        sep3 = (gammaRecj * psumRec + gammaTrij * psumTri) / mindni
        
        
        return (sep1 + sep2 + sep3) / ((1 + d_i) ** 0.5)

    def _debug_PARic(self, i, j, beta):
        
        # Ric(i, j) = 1/sq(d_i+1) * ((1-P_ji)/(d_j+1)+psumTri/beta+(gammaRecj*psumRec+gammaTrij*psumTri)/mindn_i)
        d_i = self.d(i)
        d_j = self.d(j)
        psumTri = self.psum_Triangle(i, j)
        gammaTrij = self.gammaTri(i, j)
        psumRec = self.psum_Rectangle(i, j)
        gammaRecj = self.gammaRec(i, j)
        mindni = self.mindn(i)
        
        sep1 = (1 - self.W[j, i]) / (d_j + 1)
        
        sep2 = psumTri / beta
        
        sep3 = (gammaRecj * psumRec + gammaTrij * psumTri) / mindni
        
        ric = (sep1 + sep2 + sep3) / ((1 + d_i) ** 0.5)
        
        print(f"d_i:{d_i}, d_j:{d_j}, psumTri:{psumTri}, gammaTri:{gammaTrij}, psumRec:{psumRec}, gammaRecj:{gammaRecj}, mindni:{mindni}")
        print(f"ric:{ric}, sep1:{sep1}, sep2:{sep2}, sep3:{sep3}")
        
        
        return ric
    
    def mapRic(self, method="PARic", beta=1):
        if method == "PARic":
            func = self.PARic
        elif method == "BFRic":
            func = self.BFRic
        
        map = np.zeros([self.num_V, self.num_V]) - np.inf
        
        #for i in tqdm(range(self.num_V)):
        for i in range(self.num_V):
            for j in range(self.num_V):
                if i != j and self.A[i, j] != 0:
                    map[i, j] = func(i, j, beta)
            #print(i, j)
        
        return map
    
    def _reachability(self):
        # count how many edges fitts
        # Pjk
        
        def averageD():
            aveD = 0
            for i in range(self.num_V):
                aveD += (self.d(i) + 1) ** -0.5
            return aveD / self.num_V
        
        aveD = averageD()
        mapCurv = self.mapRic()
        edges = 0
        reaches = 0
        
        for i in range(self.numV):
            for j in range(self.num_V):
                if i != j and self.A[i, j] != 0:
                    edges += 1



def skip_connection(adj, ric, kappa, edge_index, edge_type):

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if i != j and ric[i, j] != -np.inf and ric[i, j] < kappa:
                for k in range(adj.shape[1]):
                    if adj[j][k] > 0 and k != i:
                        edge_index = add_edge(edge_index, i, k)
                        edge_type = np.append(edge_type, 1)
    
    return edge_index, edge_type

def dense_adj_rw_normalize(adj):
    """Random-walk normalize adjacency matrix and convert to dense"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv = np.power(rowsum, -1).flatten() # D^-1
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv) # D^-1
    return adj.dot(d_mat_inv).transpose() # D^-1A   

def rewiring(edge_index, beta, kappa, edge_type=None):
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    

    if edge_type is None:
        edge_type = np.zeros(m, dtype=np.int64)
    Adj = torch_geometric.utils.to_scipy_sparse_matrix(torch.tensor(edge_index))
    Adj = Adj.todense()
    Adj = np.array(Adj)
    
    AdjRW = dense_adj_rw_normalize(Adj)
    AdjRW = np.array(AdjRW.todense())
    Model = PARicci2(Adj, AdjRW)
    map = Model.mapRic(beta=beta)
    
    edge_index, edge_type = skip_connection(Adj, map, kappa=kappa, edge_index=edge_index, edge_type=edge_type)
    return edge_index, edge_type
    Adj2 = skip_connection(Adj, map, kappa=kappa)
    new_edges = np.sum(Adj2)
    
    Adj12 = Adj + Adj2
    edge_index_new = sp.coo_matrix(Adj12)
    values = edge_index_new.data
    indices = np.vstack((edge_index_new.row, edge_index_new.col))
    edge_index_new = torch.LongTensor(indices)
    
def BORF_reachability1(edge_index, beta, kappa, edge_type=None):
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    
    if edge_type is None:
        edge_type = np.zeros(m, dtype=np.int64)
    A = np.array(torch_geometric.utils.to_scipy_sparse_matrix(torch.tensor(edge_index)).todense())
    print(A.shape)
    print((A - A.T).sum())
    edges = 0
    avail = 0
    
    for i in range(A.shape[0]):
        for j in range(i, A.shape[0]):
            if A[i,j]:
                # any edge
                edges += 1
                flag = True
                maxn = max(A[i].sum(), A[j].sum())
                minn = min(A[i].sum(), A[j].sum())
                ratio = maxn / minn
                
                for k in  np.argwhere(A[i]>0):
                    connect = 0
                    for w in np.argwhere(A[j]>0):
                        if k==w or A[k,w]:
                            connect += 1
                    if connect>ratio:
                        flag = False
                        break
                for w in  np.argwhere(A[i]>0):
                    connect = 0
                    for k in np.argwhere(A[j]>0):
                        if k==w or A[k,w]:
                            connect += 1
                    if connect>ratio:
                        flag = False
                        break
                if flag:
                    avail += 1
    print(f"--{avail}/{edges}--")
    

def BORF_reachability2(edge_index, beta, kappa, edge_type=None):
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    
    if edge_type is None:
        edge_type = np.zeros(m, dtype=np.int64)
    A = np.array(torch_geometric.utils.to_scipy_sparse_matrix(torch.tensor(edge_index)).todense())
    print(A.shape)
    print((A - A.T).sum())
    edges = 0
    avail = 0
    
    for i in range(A.shape[0]):
        for j in range(i, A.shape[0]):
            if A[i,j]:
                # any edge
                edges += 1
                flag = True
                maxn = max(A[i].sum(), A[j].sum())
                minn = min(A[i].sum(), A[j].sum())
                ratio = maxn / minn
                if maxn != minn:
                    flag = False
                
                for k in  np.argwhere(A[i]>0):
                    if A[k].sum() != maxn:
                        flag = False
                        break
                    connect = 0
                    for w in np.argwhere(A[j]>0):
                        if k==w or A[k,w]:
                            connect += 1
                    if connect>ratio:
                        flag = False
                        break
                for w in  np.argwhere(A[i]>0):
                    if A[w].sum() != maxn:
                        flag = False
                        break
                    connect = 0
                    for k in np.argwhere(A[j]>0):
                        if k==w or A[k,w]:
                            connect += 1
                    if connect>ratio:
                        flag = False
                        break
                if flag:
                    
                    avail += 1
    print(f"--{avail}/{edges}--")
    
    
    
