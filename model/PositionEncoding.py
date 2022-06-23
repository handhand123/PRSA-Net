from utils import get_pos

class Region_PositionEncoding():
    def __init__(self, temporal_scale, dataset_name):

        self.get_connections(temporal_scale)
        self.pos = self.get_adjacency_mat(temporal_scale, dataset_name)

    def get_connections(self, num_node):
        self.I = [(i, i) for i in range(num_node)]
        self.in_neighbor = [(i - 1, i) for i in range(1, num_node)] + [(num_node - 1, 0)]
        self.out_neighbor = [(i, i - 1) for i in range(num_node - 1, 0, -1)] + [(0, num_node - 1)]
        self.neighbor = self.in_neighbor + self.out_neighbor

    def get_adjacency_mat(self, num_node, dataset_name):
        if dataset_name == 'thumos':
            pass
        elif dataset_name == 'acnet':
            assert num_node == 100
        pos = get_pos(num_node, self.I, self.in_neighbor, self.out_neighbor)
        return pos