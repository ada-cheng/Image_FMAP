import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import torch
 #from matplotlib import pyplot as plt
EPS = 1e-8

class Image_Graph():
    def __init__(self, in_image_feat=None, *args):
        """
        in_image_feat: (16,16,384)
        args[0] : 'r_radius_connected' or '2d_grid' or a Graph object
        future todos:
        1. r_radius_connected: randomly connected graph with radius r
        2. create instance from a Graph object
        """
        self.graph = None

        if in_image_feat is not None:
            if isinstance(args[0], str):
                h, w,_= in_image_feat.shape
                if args[0] == 'r_radius_connected':
                    radius = args[1]
                    self.graph = nx.random_geometric_graph(w, radius, dim=2)
                elif args[0] == '2d_grid':
                    self.graph = nx.grid_2d_graph(h, w)
                    self.adjust_weights_via_feature_differences(in_image_feat, 'normalized_cut')
                else:
                    raise ValueError("args[0] must be either 'r_radius_connected' or '2d_grid'.")
            elif isinstance(args[0], nx.Graph):
                """
                simply copy from the original code, does not have practical usages now
                """
                self.graph = args[0].__class__
                self.graph.add_nodes_from(args[0].nodes(data=True))
                self.graph.add_edges_from(args[0].edges(data=True))
            else:
                raise ValueError("args[0] must be either a string specifying a graph or a Graph object.")
            
       

    def adjust_weights_via_feature_differences(self, features, recipe, **kwargs):
        """
        assume features (16,16,384)
        recipe: 'normalized_cut'
        weight = exp(-||f1-f2||^2/sigma_f) * exp(-||p1-p2||^2/sigma_s)
        p1-p2 is spatial distance, equals 0 or 1
        """
        h, w, _ = features.shape

        if recipe == 'normalized_cut':
            sigma_f = kwargs.get('sigma_f', 'median')
            sigma_s = kwargs.get('sigma_s', 'median')
            # feature distance
            edges = list(self.graph.edges())
            feat_values = []
            spatial_values = []
            for edge in edges: # (x1,y1) (x2,y2)
                f_from = features[edge[0][0], edge[0][1], :] 
                f_to = features[edge[1][0], edge[1][1], :]
                feature_dist = np.linalg.norm(f_from - f_to) 
                feat_values.append(feature_dist)
                spatial_values.append(1)
            # calculate feature distance
            sigma_f = check_and_derive_sigma(sigma_f, feat_values) + EPS
            feat_values = np.exp(-np.square(feat_values) / sigma_f)

            # calculate spatial distance
            sigma_s = check_and_derive_sigma(sigma_s, spatial_values) + EPS
            spatial_dists = np.exp(-np.square(spatial_values) / sigma_s)
        
            
            feat_values = np.multiply(feat_values, spatial_dists)
            edges_with_weight = [(edge[0],edge[1],weight) for edge,weight in zip(edges,feat_values)]
            
            self.graph.add_weighted_edges_from(edges_with_weight)
            adj_matrix = nx.adjacency_matrix(self.graph)
            # check the number of non-zero elements
           
            
        else:
            raise NotImplementedError("Not implemented yet.")

 
def check_and_derive_sigma(sigma, values):
    if sigma == 'median':
        return 2 * (np.median(values)**2)
    else:
        if sigma <= 0:
            raise ValueError("Provided standard deviation parameters must be all positive.")
        return sigma

    
    
if __name__ == "__main__":
    pass