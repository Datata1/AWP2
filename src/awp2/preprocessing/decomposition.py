"""Decomposition for the shared preprocessing pipeline."""

from typing import Self

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.neural_network import BernoulliRBM

from awp2.config import (
    LPP_RANK_RELATIVE_TOLERANCE,
    PCA_FEATURE_NAMES,
    SEED,
)
from awp2.preprocessing.bands import _BandTransformer


class OrthogonalSubspaceProjection(_BandTransformer):
    """Unsupervised OSP by automatic target generation: iterative endmember signatures.

    Each round projects the training spectra onto the orthogonal complement of the
    endmembers found so far and picks the highest-energy pixel as the next endmember.
    New spectra are projected onto the endmembers. Runs on raw spectra, so it must not
    combine with scaling.
    """

    def __init__(self, n_components: int) -> None:
        """Set the number of endmembers.

        Args:
            n_components: Number of endmember signatures to extract.
        """
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Extract endmembers greedily from the training spectra.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the component count is not positive or exceeds the bands.
        """
        X = self._check(X)
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.n_components > X.shape[1]:
            raise ValueError("n_components must not exceed the bands")
        self._set_input(X)
        values = X.to_numpy(dtype=float)
        found = np.zeros((X.shape[1], 0))
        for _ in range(self.n_components):
            if found.shape[1]:
                projector = np.eye(X.shape[1]) - found @ np.linalg.pinv(found)
            else:
                projector = np.eye(X.shape[1])
            energies = ((values @ projector) ** 2).sum(axis=1)
            found = np.column_stack([found, values[np.argmax(energies)]])
        self.endmembers_ = found
        self.feature_names_out_ = [f"osp{position}" for position in range(self.n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project spectra onto the fitted endmembers.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            Endmember projections, one ``osp<i>`` column each.
        """
        values = self._check(X).to_numpy(dtype=float, copy=True)
        return pd.DataFrame(
            values @ self.endmembers_, index=X.index, columns=self.feature_names_out_
        )


class VerySparseRandomProjection(_BandTransformer):
    """Achlioptas very sparse random projection of mean-centered spectra.

    Entries are ``±sqrt(c)`` with probability ``1/(2c)`` each and zero otherwise, with
    ``c`` as the square root of the band count. The projection is data-independent; the
    training mean is the only fitted statistic, so centering holds for new spectra too.
    """

    def __init__(self, n_components: int) -> None:
        """Set the output size.

        Args:
            n_components: Number of random components.
        """
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Store the training mean and draw the fixed projection matrix.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the component count is not positive.
        """
        X = self._check(X)
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        self._set_input(X)
        self.mean_ = X.to_numpy(dtype=float).mean(axis=0)
        sparsity = float(np.sqrt(X.shape[1]))
        draw = np.random.default_rng(SEED).random((X.shape[1], self.n_components))
        self.matrix_ = np.where(
            draw < 1 / (2 * sparsity), 1.0, np.where(draw < 1 / sparsity, -1.0, 0.0)
        ) * np.sqrt(sparsity)
        self.feature_names_out_ = [f"vsrp{position}" for position in range(self.n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Center spectra with the training mean and apply the fixed projection.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            Random components, one ``vsrp<i>`` column each.
        """
        values = self._check(X).to_numpy(dtype=float, copy=True)
        return pd.DataFrame(
            (values - self.mean_) @ self.matrix_, index=X.index, columns=self.feature_names_out_
        )


class LocalityPreservingProjection(_BandTransformer):
    """Linear locality-preserving reduction of the spectral bands.

    Builds a symmetric binary k-nearest-neighbour graph over the training spectra and keeps
    the projection that best preserves neighbour relations (smallest eigenvectors of the
    generalized problem ``XLX'a = lXDX'a``). New spectra transform linearly, so validation
    data needs no graph of its own. Numerically null directions of the degree-weighted
    spectral covariance are excluded before solving the eigenproblem.
    """

    def __init__(self, n_components: int, n_neighbors: int) -> None:
        """Set the output size and the neighbourhood size.

        Args:
            n_components: Number of preserved components.
            n_neighbors: Neighbours per spectrum for the adjacency graph.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Build the neighbourhood graph and solve for the preserving projection.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the component or neighbour counts are not positive, or if the
                component count exceeds the numerical rank of the weighted training spectra.
        """
        X = self._check(X)
        if self.n_components < 1 or self.n_neighbors < 1:
            raise ValueError("n_components and n_neighbors must be positive")
        self._set_input(X)
        values = X.to_numpy(dtype=float)
        graph = kneighbors_graph(values, self.n_neighbors, include_self=False)
        adjacency = graph.maximum(graph.T)
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        spread = values * degrees[:, None] - adjacency @ values
        left = spread.T @ values
        right = (values * degrees[:, None]).T @ values
        eigenvalues, basis = eigh((right + right.T) / 2)
        retained = eigenvalues > eigenvalues.max() * LPP_RANK_RELATIVE_TOLERANCE
        self.effective_rank_ = int(np.count_nonzero(retained))
        if self.n_components > self.effective_rank_:
            raise ValueError(
                f"n_components={self.n_components} exceeds the numerical rank "
                f"of the weighted training spectra ({self.effective_rank_})"
            )
        whitening = basis[:, retained] / np.sqrt(eigenvalues[retained])
        reduced = whitening.T @ left @ whitening
        _, eigenvectors = eigh(
            (reduced + reduced.T) / 2, subset_by_index=[0, self.n_components - 1]
        )
        self.projection_ = whitening @ eigenvectors
        self.feature_names_out_ = [f"lpp{position}" for position in range(self.n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project spectra onto the fitted preserving components.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            Preserved components, one ``lpp<i>`` column each.
        """
        values = self._check(X).to_numpy(dtype=float, copy=True)
        return pd.DataFrame(
            values @ self.projection_, index=X.index, columns=self.feature_names_out_
        )


class DeepBeliefFeatures(_BandTransformer):
    """DBN-style features from greedily stacked BernoulliRBMs.

    Each layer is trained on the hidden probabilities of the previous one. Visible units
    assume ``[0, 1]`` input, hence the min-max scaler requirement.
    """

    def __init__(self, hidden_layers: tuple[int, ...]) -> None:
        """Set the stacked architecture.

        Args:
            hidden_layers: Hidden units per layer; the last entry is the output size.
        """
        self.hidden_layers = hidden_layers

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Train the stacked machines greedily, bottom layer first.

        Args:
            X: Band columns of the training part, scaled to ``[0, 1]``.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the architecture is empty or holds non-positive sizes.
        """
        X = self._check(X)
        if not self.hidden_layers or any(size < 1 for size in self.hidden_layers):
            raise ValueError("hidden_layers must hold at least one positive size")
        self._set_input(X)
        activations = X.to_numpy(dtype=float)
        self.rbms_ = []
        for size in self.hidden_layers:
            machine = BernoulliRBM(n_components=size, random_state=SEED)
            activations = machine.fit_transform(activations)
            self.rbms_.append(machine)
        self.feature_names_out_ = [f"dbn{position}" for position in range(self.hidden_layers[-1])]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Propagate spectra through the stacked machines.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            Top-layer hidden probabilities, one ``dbn<i>`` column each.
        """
        activations = self._check(X).to_numpy(dtype=float, copy=True)
        for machine in self.rbms_:
            activations = machine.transform(activations)
        return pd.DataFrame(activations, index=X.index, columns=self.feature_names_out_)


class PrincipalComponentFeatures(_BandTransformer):
    """Reduce standardized spectral bands to the two named PCA features."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Fit PCA to the standardized training spectra.

        Args:
            X: Standardized band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.
        """
        X = self._check(X)
        self._set_input(X)
        self.pca_ = PCA(n_components=len(PCA_FEATURE_NAMES), random_state=SEED).fit(X)
        self.feature_names_out_ = list(PCA_FEATURE_NAMES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project spectra onto the fitted principal-component axes.

        Args:
            X: Standardized band columns with the same columns as in ``fit``.

        Returns:
            One ``PC_1`` and one ``PC_2`` feature per spectrum.
        """
        values = self.pca_.transform(self._check(X))
        return pd.DataFrame(values, index=X.index, columns=self.feature_names_out_)
