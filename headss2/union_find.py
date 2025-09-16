class UnionFind:
    """Union-find algorithm to sequentially merge overlapping clusters"""

    def __init__(self):
        self.parent = {}

    def find(self, cluster: int) -> int:
        """Get root of the set of 'cluster'"""
        if cluster not in self.parent:
            self.parent[cluster] = cluster
        if self.parent[cluster] != cluster:
            self.parent[cluster] = self.find(self.parent[cluster])
        return self.parent[cluster]
    
    def union(self, cluster1: int, cluster2: int) -> None:
        """Merge two clusters"""
        self.parent[self.find(cluster1)] = self.find(cluster2)