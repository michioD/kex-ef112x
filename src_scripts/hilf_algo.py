import numpy as np
import bisect

class HIL_F:
    def __init__(self, n_samples, beta=0.5):
        self.boundaries = [0.0, 1.0]
        self.weights = [1.0]
        self.beta = beta
        self.n_samples = n_samples
        self.eta = np.sqrt(8 * np.log(n_samples + 1) / n_samples)
        self.weight_history = [list(self.weights)]
        self.boundary_history = [list(self.boundaries)]
        self.history = []

    def get_decision(self, p_t):
        """
        Returns a probabilistic decision using the continuous weight area method.
        """
        area_below = 0.0
        total_area = 0.0

        for i in range(len(self.weights)):
            w = self.weights[i]
            b_low = self.boundaries[i]
            b_high = self.boundaries[i + 1]

            area = w * (b_high - b_low)
            total_area += area

            if b_high <= p_t:
                area_below += area
            elif b_low < p_t < b_high:
                area_below += w * (p_t - b_low)

        q_t = area_below / total_area if total_area > 0 else 0.5
        q_t = np.clip(q_t, 0.0, 1.0)

        accept_sml = np.random.rand() < q_t
        return accept_sml, q_t

    def update(self, p_t, y_t):
        """
        Updates boundaries and weights using continuous exponential updates.
        """
        # Split interval at p_t to allow for more granular learning
        if not np.any(np.isclose(self.boundaries, p_t, atol=1e-8)):
            idx = bisect.bisect_left(self.boundaries, p_t)
            self.boundaries.insert(idx, p_t)
            self.weights.insert(idx, self.weights[idx - 1])
        
        # Identify the index where boundaries[idx] == p_t
        # Note: boundaries[0] is 0.0, so the first weight corresponds to [0, p_t]
        split_idx = bisect.bisect_left(self.boundaries, p_t)

        # Exponential weight update
        for i in range(len(self.weights)):
            # If the interval boundary is below p_t, this expert 'would have' accepted.
            if self.boundaries[i+1] <= p_t:
                loss = y_t
            # If the interval boundary is above p_t, this expert 'would have' offloaded.
            else:
                loss = self.beta
            
            self.weights[i] *= np.exp(-self.eta * loss)
            
        # Prevent numerical underflow by periodically normalizing the weights
        # We divide by max(weights) so that the 'best' expert is always at 1.0
        max_w = max(self.weights)
        if max_w < 1e-100 or max_w > 1e10:
            self.weights = [w / max_w for w in self.weights]

        self.weight_history.append(list(self.weights))
        self.boundary_history.append(list(self.boundaries))
        self.history.append((p_t, y_t))
