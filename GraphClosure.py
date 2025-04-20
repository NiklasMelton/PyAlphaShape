class GraphClosureTracker:
    def __init__(self, num_nodes):
        # Initialize each node to be its own parent (self-loop)
        self.parent = list(range(num_nodes))
        self.rank = [0] * num_nodes  # Rank array to optimize union operation
        self.num_nodes = num_nodes
        self.components = {i: {i} for i in range(num_nodes)}  # Initial components

    def _ensure_capacity(self, node: int):
        if node >= self.num_nodes:
            # extend parent / rank arrays and component dict
            for i in range(self.num_nodes, node + 1):
                self.parent.append(i)
                self.rank.append(0)
                self.components[i] = {i}
            self.num_nodes = node + 1

    # modify the public methods to call it
    def find(self, node):
        self._ensure_capacity(node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        # Union by rank and update components
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 != root2:
            # Attach smaller rank tree under the larger rank tree
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
                # Update the components by merging sets
                self.components[root1].update(self.components[root2])
                del self.components[root2]
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
                self.components[root2].update(self.components[root1])
                del self.components[root1]
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
                self.components[root1].update(self.components[root2])
                del self.components[root2]

    def add_edge(self, node1, node2):
        # Add an edge by connecting two nodes
        self.union(node1, node2)

    def add_fully_connected_subgraph(self, nodes):
        # Connect each pair of nodes to form a fully connected subgraph
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.union(nodes[i], nodes[j])

    def subgraph_is_already_connected(self, nodes):
        # Check if all nodes in the given list are connected

        if not nodes:
            return True  # Empty list is trivially connected
        # Find the root of the first node
        root = self.find(nodes[0])
        # Check if all other nodes share this root
        return all(self.find(node) == root for node in nodes)

    def is_connected(self, node1, node2):
        # Check if two nodes are in the same component
        return self.find(node1) == self.find(node2)

    def __iter__(self):
        # Make the class iterable over connected components
        return iter(self.components.values())

    def __getitem__(self, index):
        # Make the class indexable over connected components
        return list(self.components.values())[index]

    def __len__(self):
        # Return the number of connected components
        return len(self.components)