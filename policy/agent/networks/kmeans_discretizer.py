import tqdm
import torch
from sklearn.cluster import KMeans

class KMeansDiscretizer:
	"""
	Simplified and modified version of KMeans algorithm from sklearn.

	Code borrowed from https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
	"""

	def __init__(
		self,
		num_bins: int = 100,
		kmeans_iters: int = 50,
	):
		super().__init__()
		self.n_bins = num_bins
		self.kmeans_iters = kmeans_iters

	def fit(self, input_actions: torch.Tensor) -> None:
		self.bin_centers = KMeansDiscretizer._kmeans(
			input_actions, nbin=self.n_bins, niter=self.kmeans_iters
		)

	@classmethod
	def _kmeans(cls, x: torch.Tensor, nbin: int = 512, niter: int = 50):
		"""
		Function implementing the KMeans algorithm.

		Args:
			x: torch.Tensor: Input data - Shape: (N, D)
			nbin: int: Number of bins
			niter: int: Number of iterations
		"""

		# TODO: Implement KMeans algorithm to cluster x into nbin bins. Return the bin centers - shape (nbin, x.shape[-1])
		x_np = x.cpu().numpy()
		kmeans = KMeans(n_clusters=nbin, max_iter = niter, random_state=0, n_init="auto").fit(x_np)
		bin_centers = torch.tensor(kmeans.cluster_centers_)
		return bin_centers
