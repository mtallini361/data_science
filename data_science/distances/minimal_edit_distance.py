
class MinimalEditDistance:
    def __init__(self, deletion_cost=1, insertion_cost=1, substitution_cost=2):
        self.deletion_cost = deletion_cost
        self.insertion_cost = insertion_cost
        self.substitution_cost = substitution_cost

    def measure(self, source, target):
        n = len(source)
        m = len(target)
        
        # Initialize distance matrix (size (n+1) x (m+1))
        D = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Initialization: distance from empty string
        D[0][0] = 0
        for i in range(1, n + 1):
            D[i][0] = D[i - 1][0] + self.deletion_cost
        for j in range(1, m + 1):
            D[0][j] = D[0][j - 1] + self.insertion_cost
        
        # Fill in the matrix using recurrence
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i][j] = min(
                    D[i - 1][j] + self.deletion_cost,  # deletion
                    D[i - 1][j - 1] + (self.substitution_cost if source[i - 1] != target[j - 1] else 0),  # substitution
                    D[i][j - 1] + self.insertion_cost # insertion
                )
        
        # Termination: return bottom-right value
        return D[n][m]