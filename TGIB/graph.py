import numpy as np
import torch

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
       
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform
        
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0] 

        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr]) 
            e_idx_l.extend([x[1] for x in curr]) 
            n_ts_l.extend([x[2] for x in curr]) 
            off_set_l.append(len(n_idx_l)) 

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, off_set_l
        
    def find_before(self, src_idx, cut_time): 
        """
        Params
        ------
        src_idx: int
        cut_time: float
        """
        start = self.off_set_l[src_idx]
        end = self.off_set_l[src_idx + 1]

        if start == end:
            return self.node_idx_l[start:end], self.edge_idx_l[start:end], self.node_ts_l[start:end]

        neighbors_ts = self.node_ts_l[start:end]
        n_valid = np.searchsorted(neighbors_ts, cut_time, side='left')

        return (self.node_idx_l[start:start + n_valid],
                self.edge_idx_l[start:start + n_valid],
                self.node_ts_l[start:start + n_valid])

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: numpy array of int
        cut_time_l: numpy array of float
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        B = len(src_idx_l)
        out_ngh_node_batch = np.zeros((B, num_neighbors), dtype=np.int32) 
        out_ngh_t_batch = np.zeros((B, num_neighbors), dtype=np.float32) 
        out_ngh_eidx_batch = np.zeros((B, num_neighbors), dtype=np.int32) 

        off = self.off_set_l
        node_idx = self.node_idx_l
        node_ts = self.node_ts_l
        edge_idx = self.edge_idx_l

        for i in range(B):
            src_idx = src_idx_l[i]
            cut_time = cut_time_l[i]

            start = off[src_idx]
            end = off[src_idx + 1]

            if start == end:
                continue

            n_valid = np.searchsorted(node_ts[start:end], cut_time, side='left')

            if n_valid == 0:
                continue

            if self.uniform:
                sample_count = min(n_valid, num_neighbors)
                sampled_idx = np.random.randint(0, n_valid, sample_count)
                out_ngh_node_batch[i, num_neighbors - sample_count:] = node_idx[start:end][sampled_idx]
                out_ngh_t_batch[i, num_neighbors - sample_count:] = node_ts[start:end][sampled_idx]
                out_ngh_eidx_batch[i, num_neighbors - sample_count:] = edge_idx[start:end][sampled_idx]

                pos = out_ngh_t_batch[i, num_neighbors - sample_count:].argsort()
                out_ngh_node_batch[i, num_neighbors - sample_count:] = out_ngh_node_batch[i, num_neighbors - sample_count:][pos]
                out_ngh_t_batch[i, num_neighbors - sample_count:] = out_ngh_t_batch[i, num_neighbors - sample_count:][pos]
                out_ngh_eidx_batch[i, num_neighbors - sample_count:] = out_ngh_eidx_batch[i, num_neighbors - sample_count:][pos]
            else:
                take_count = min(n_valid, num_neighbors)
                out_ngh_node_batch[i, num_neighbors - take_count:] = node_idx[start:start + take_count]
                out_ngh_t_batch[i, num_neighbors - take_count:] = node_ts[start:start + take_count]
                out_ngh_eidx_batch[i, num_neighbors - take_count:] = edge_idx[start:start + take_count]

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch 


    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k -1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1] 
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors) 
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records
