from scipy.sparse import csr_matrix
import torch
#from matplotlib import pyplot as plt
import torch_geometric.datasets 
import torch_geometric.utils 
import networkx as nx
class Image_Graph():
    def __init__(self, in_image_feat=None,*args):
        """
        in_image_feat: (h,w,384)
        args[0] : 'r_radius_connected' or '2d_grid' or a Graph object
        """
        self.graph = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != in_image_feat.device.type:
            raise ValueError("in_image_feat must be on the same device as the graph.")
        if in_image_feat is not None:
            if isinstance(args[0], str):
                w, h ,_= in_image_feat.shape
                if args[0] == 'r_radius_connected':
                    radius = args[1]
                    raise ValueError("Unimplemented")
                elif args[0] == '2d_grid':
                    temp_graph = nx.grid_2d_graph(w,h)
                    temp_int = nx.convert_node_labels_to_integers(temp_graph)
                    edge_index = torch.tensor(list(temp_int.edges()), device = self.device).t().contiguous()
                    self.graph = torch_geometric.data.Data(edge_index=edge_index)
                    use_dino = args[1] if len(args) > 1 else True
                    self.adjust_weights_via_feature_differences(in_image_feat, 'normalized_cut', use_dino = use_dino)
                else:
                    raise ValueError("args[0] must be either 'r_radius_connected' or '2d_grid'.")
            elif isinstance(args[0], nx.Graph):
                temp_int = nx.convert_node_labels_to_integers(args[0])
                edge_index = torch.tensor(list(temp_int.edges()), device = device).t().contiguous()
                self.graph = torch_geometric.data.Data(edge_index=edge_index)
            else:
                raise ValueError("args[0] must be either a string specifying a graph or a Graph object.")
        
                    
    def adjust_weights_via_feature_differences(self, features, recipe, use_dino = True,**kwargs):
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
            num_nodes = self.graph.num_nodes
            if use_dino:
                features = features.reshape((num_nodes, -1))
                f_from = features[self.graph.edge_index[0]]
                f_to = features[self.graph.edge_index[1]]
                feat_values = torch.norm(f_from - f_to, dim=1)
            else:
                feat_values = torch.ones((self.graph.edge_index.shape[-1],), device = self.device)
            spatial_values = torch.ones((self.graph.num_edges), device = self.device)
            sigma_f = check_and_derive_sigma(sigma_f, feat_values)
            feat_values = torch.exp(-torch.square(feat_values) / sigma_f)
            sigma_s = check_and_derive_sigma(sigma_s, spatial_values)
            spatial_values = torch.exp(-torch.square(spatial_values) / sigma_s)
            feat_values = torch.mul(feat_values, spatial_values)
            self.graph.edge_attr = feat_values
        else:
            raise ValueError("Not implemented yet.")
            
            
 
def check_and_derive_sigma(sigma, values):
    if sigma == 'median':
        return 2 * (torch.median(values) ** 2)
    else:
        if sigma <= 0:
            raise ValueError("Provided standard deviation parameters must be all positive.")
        return sigma


