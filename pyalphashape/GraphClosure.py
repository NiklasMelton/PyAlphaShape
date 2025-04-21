from typing import List
class GraphClosureTracker:
    """
    A dynamic Union-Find (Disjoint Set) data structure with explicit tracking
    of connected components and support for dynamic resizing.

    Useful for managing dynamic connectivity in undirected graphs.
    """

    def __init__(self, num_nodes: int):
        """
        Initialize the tracker with a specified number of nodes.

        Parameters
        ----------
        num_nodes : int
            The initial number of nodes in the graph.
        """

        self.parent = list(range(num_nodes))
        self.rank = [0] * num_nodes  # Rank array to optimize union operation
        self.num_nodes = num_nodes
        self.components = {i: {i} for i in range(num_nodes)}  # Initial components


    def _ensure_capacity(self, node: int) -> None:
        """
        Dynamically expand internal arrays to include a node with the given index.

        Parameters
        ----------
        node : int
            The node index to ensure capacity for.
        """

        if node >= self.num_nodes:
            # extend parent / rank arrays and component dict
            for i in range(self.num_nodes, node + 1):
                self.parent.append(i)
                self.rank.append(0)
                self.components[i] = {i}
            self.num_nodes = node + 1

    # modify the public methods to call it
    def find(self, node: int) -> int:
        """
        Find the root representative of the set containing the node.

        Parameters
        ----------
        node : int
            The node whose component root is to be found.

        Returns
        -------
        int
            The root node of the component.
        """

        self._ensure_capacity(node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1: int, node2: int) -> None:
        """
        Merge the components containing node1 and node2.

        Parameters
        ----------
        node1 : int
            First node.
        node2 : int
            Second node.
        """

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

    def add_edge(self, node1: int, node2: int) -> None:
        """
        Add an undirected edge between two nodes by merging their components.

        Parameters
        ----------
        node1 : int
            First node.
        node2 : int
            Second node.
        """

        self.union(node1, node2)

    def add_fully_connected_subgraph(self, nodes: List[int]) -> None:
        """
        Fully connect a list of nodes by merging all pairs into one component.

        Parameters
        ----------
        nodes : List[int]
            A list of node indices to be fully connected.
        """

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.union(nodes[i], nodes[j])

    def subgraph_is_already_connected(self, nodes: List[int]) -> bool:
        """
        Check whether all nodes in the list belong to the same connected component.

        Parameters
        ----------
        nodes : List[int]
            A list of node indices.

        Returns
        -------
        bool
            True if all nodes are connected, False otherwise.
        """

        if not nodes:
            return True  # Empty list is trivially connected
        # Find the root of the first node
        root = self.find(nodes[0])
        # Check if all other nodes share this root
        return all(self.find(node) == root for node in nodes)

    def is_connected(self, node1: int, node2: int) -> bool:
        """
        Check whether two nodes are in the same connected component.

        Parameters
        ----------
        node1 : int
            First node.
        node2 : int
            Second node.

        Returns
        -------
        bool
            True if node1 and node2 are connected, False otherwise.
        """

        return self.find(node1) == self.find(node2)

    def __iter__(self):
        """
        Iterate over the current connected components.

        Returns
        -------
        Iterator[Set[int]]
            An iterator over sets of node indices.
        """

        return iter(self.components.values())

    def __getitem__(self, index: int) -> List[int]:
        """
        Index into the list of connected components.

        Parameters
        ----------
        index : int
            Index of the connected component to access.

        Returns
        -------
        List[int]
            List of nodes in the selected connected component.
        """

        return list(self.components.values())[index]

    def __len__(self) -> int:
        """
        Return the number of connected components.

        Returns
        -------
        int
            The number of components currently being tracked.
        """

        return len(self.components)