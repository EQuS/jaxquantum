"""Measurements."""

import optax
import jax.numpy as jnp

from collections.abc import Callable
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Optional, NamedTuple
from functools import partial, reduce

from jax import config, Array, jit, value_and_grad, lax, vmap

from jaxquantum.core.qarray import Qarray, powm
from jaxquantum.core.operators import identity

from jax_tqdm import scan_tqdm

config.update("jax_enable_x64", True)


# Calculations ----------------------------------------------------------------


def overlap(rho: Qarray, sigma: Qarray) -> Array:
    """Overlap between two states or operators.

    Args:
        rho: state/operator.
        sigma: state/operator.

    Returns:
        Overlap between rho and sigma.
    """

    if rho.is_vec() and sigma.is_vec():
        return jnp.abs(((rho.to_ket().dag() @ sigma.to_ket()).trace())) ** 2
    elif rho.is_vec():
        rho = rho.to_ket()
        res = (rho.dag() @ sigma @ rho).data
        return res.squeeze(-1).squeeze(-1)
    elif sigma.is_vec():
        sigma = sigma.to_ket()
        res = (sigma.dag() @ rho @ sigma).data
        return res.squeeze(-1).squeeze(-1)
    else:
        return (rho.dag() @ sigma).trace()


def fidelity(rho: Qarray, sigma: Qarray, force_positivity: bool=False) -> (
        jnp.ndarray):
    """Fidelity between two states.

    Args:
        rho: state.
        sigma: state.
        force_positivity: force the states to be positive semidefinite

    Returns:
        Fidelity between rho and sigma.
    """
    rho = rho.to_dm()
    sigma = sigma.to_dm()

    sqrt_rho = powm(rho, 0.5, clip_eigvals=force_positivity)

    return jnp.real(((powm(sqrt_rho @ sigma @ sqrt_rho, 0.5,
                           clip_eigvals=force_positivity)).tr())
                    ** 2)


def _reconstruct_density_matrix(params: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Pure function to parameterize a density matrix.
    Ensures the resulting matrix is positive semi-definite and has trace 1.
    """
    num_real_params = dim * (dim + 1) // 2

    real_part_flat = params[:num_real_params]
    imag_part_flat = params[num_real_params:]

    T = jnp.zeros((dim, dim), dtype=jnp.complex128)

    # Set the real parts of the lower triangle from the first part of params
    tril_indices = jnp.tril_indices(dim)
    T = T.at[tril_indices].set(real_part_flat)

    # Set the imaginary parts of the strictly lower triangle from the second part of params
    tril_indices_off_diag = jnp.tril_indices(dim, k=-1)
    T = T.at[tril_indices_off_diag].add(1j * imag_part_flat)

    rho_unnormalized = T @ T.conj().T
    # Enforce trace=1 by dividing by the trace
    trace = jnp.trace(rho_unnormalized)
    return rho_unnormalized / jnp.where(trace == 0, 1.0, trace)


def _parametrize_density_matrix(rho_data: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Calculates the parameter vector from a density matrix using Cholesky decomposition.
    This is the inverse of the _reconstruct_density_matrix function.
    """
    # Add a small epsilon for numerical stability, ensuring the matrix is positive definite
    T = jnp.linalg.cholesky(rho_data + 1e-9 * jnp.eye(dim))

    # T is lower-triangular with a real, positive diagonal. This matches our
    # parameterization convention.

    # Extract the real parts of all lower-triangular elements
    tril_indices = jnp.tril_indices(dim)
    real_part = T[tril_indices].real

    # Extract the imaginary parts of the strictly lower-triangular elements
    tril_indices_off_diag = jnp.tril_indices(dim, k=-1)
    imag_part = T[tril_indices_off_diag].imag

    return jnp.concatenate([real_part, imag_part])


def _L1_reg(params: jnp.ndarray) -> jnp.ndarray:
    """Pure function for L1 regularization."""
    return jnp.sum(jnp.abs(params))


def _likelihood(
    params: jnp.ndarray, dim: int, basis: jnp.ndarray, results: jnp.ndarray
) -> jnp.ndarray:
    """Compute the log-likelihood."""
    rho = _reconstruct_density_matrix(params, dim)
    expected_outcomes = jnp.real(jnp.einsum("ijk,jk->i", basis, rho))
    return -jnp.sum((expected_outcomes - results) ** 2)


# This is the core JIT-ted training loop. It is a pure function.
@partial(
    jit,
    static_argnames=[
        "dim",
        "epochs",
        "optimizer",
        "compute_infidelity",
        "L1_reg_strength",
    ],
)
def _run_tomography_scan(
    initial_params,
    initial_opt_state,
    true_rho_data,
    measurement_basis,
    measurement_results,
    dim,
    epochs,
    optimizer,
    compute_infidelity,
    L1_reg_strength,
):
    """
    A pure, JIT-compiled function that runs the entire optimization.
    Static arguments are those that define the computation graph and don't change during the run.
    """

    def loss(params):
        log_likelihood = _likelihood(
            params, dim, measurement_basis, measurement_results
        )
        regularization = L1_reg_strength * _L1_reg(params)
        return -log_likelihood + regularization

    loss_val_grad = value_and_grad(loss)

    @scan_tqdm(epochs)
    def train_step(carry, _):
        params, opt_state = carry
        loss_val, grads = loss_val_grad(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # This `if` statement is safe inside a JIT-ted function because
        # `compute_infidelity` is a "static" argument. JAX compiles a
        # separate version of the code for each value of this flag.
        if compute_infidelity:
            rho = Qarray.create(_reconstruct_density_matrix(params, dim))
            fid = fidelity(Qarray.create(true_rho_data), rho,
                           force_positivity=True)
            infidelity = 1.0 - fid
        else:
            infidelity = jnp.nan

        new_carry = (new_params, new_opt_state)
        history = {
            "loss": loss_val,
            "grads": grads,
            "params": params,
            "infidelity": infidelity,
        }
        return new_carry, history

    initial_carry = (initial_params, initial_opt_state)
    final_carry, history = lax.scan(
        train_step, initial_carry, jnp.arange(epochs), length=epochs
    )
    return final_carry, history


class MLETomographyResult(NamedTuple):
    rho: Qarray
    params_history: list
    loss_history: list
    grads_history: list
    infidelity_history: Optional[list]


class QuantumStateTomography:
    def __init__(
        self,
        rho_guess: Qarray,
        measurement_basis: Qarray,
        measurement_results: jnp.ndarray,
        complete_basis: Optional[Qarray] = None,
        true_rho: Optional[Qarray] = None,
    ):
        """
        Reconstruct a quantum state from measurement results using quantum state tomography.
        The tomography can be performed either by direct inversion or by maximum likelihood estimation.

        Args:
            rho_guess (Qarray): The initial guess for the quantum state.
            measurement_basis (Qarray): The basis in which measurements are performed.
            measurement_results (jnp.ndarray): The results of the measurements.
            complete_basis (Optional[Qarray]): The complete basis for state 
            reconstruction used when using direct inversion. 
            Defaults to the measurement basis if not provided.
            true_rho (Optional[Qarray]): The true quantum state, if known.

        """
        self.rho_guess = rho_guess.data
        self.measurement_basis = measurement_basis.data
        self.measurement_results = measurement_results
        self.complete_basis = (
            complete_basis.data
            if (complete_basis is not None)
            else measurement_basis.data
        )
        self.true_rho = true_rho
        self._result = None

    @property
    def result(self) -> Optional[MLETomographyResult]:
        return self._result

    def parameterize_density_matrix(self, params: jnp.ndarray, dim: int) -> jnp.ndarray:
        return _reconstruct_density_matrix(params, dim)

    def L1_reg(self, params: jnp.ndarray) -> jnp.ndarray:
        return _L1_reg(params)

    def likelihood(
        self, dim: int, params: jnp.ndarray, basis: jnp.ndarray, results: jnp.ndarray
    ) -> jnp.ndarray:
        return _likelihood(params, dim, basis, results)

    def quantum_state_tomography_mle(
        self, L1_reg_strength: float = 0.0, epochs: int = 10000, lr: float = 5e-3
    ) -> MLETomographyResult:
        """Perform quantum state tomography using maximum likelihood 
        estimation (MLE).

        This method reconstructs the quantum state from measurement results 
        by optimizing
        a likelihood function using gradient descent. The optimization 
        ensures the 
        resulting density matrix is positive semi-definite with trace 1.

        Args:
            L1_reg_strength (float, optional): Strength of L1 
            regularization. Defaults to 0.0.
            epochs (int, optional): Number of optimization iterations. 
            Defaults to 10000.
            lr (float, optional): Learning rate for the Adam optimizer. 
            Defaults to 5e-3.

        Returns:
            MLETomographyResult: Named tuple containing:
                - rho: Reconstructed quantum state as Qarray
                - params_history: List of parameter values during optimization
                - loss_history: List of loss values during optimization
                - grads_history: List of gradient values during optimization
                - infidelity_history: List of infidelities if true_rho was 
                provided, None otherwise
        """

        dim = self.rho_guess.shape[0]
        optimizer = optax.adam(lr)

        # Initialize parameters from the initial guess for the density matrix
        params = _parametrize_density_matrix(self.rho_guess, dim)
        opt_state = optimizer.init(params)

        compute_infidelity_flag = self.true_rho is not None

        # Provide a dummy array if no true_rho is available. It won't be used.
        true_rho_data_or_dummy = (
            self.true_rho.data
            if compute_infidelity_flag
            else jnp.empty((dim, dim), dtype=jnp.complex64)
        )

        final_carry, history = _run_tomography_scan(
            initial_params=params,
            initial_opt_state=opt_state,
            true_rho_data=true_rho_data_or_dummy,
            measurement_basis=self.measurement_basis,
            measurement_results=self.measurement_results,
            dim=dim,
            epochs=epochs,
            optimizer=optimizer,
            compute_infidelity=compute_infidelity_flag,
            L1_reg_strength=L1_reg_strength,
        )

        final_params, _ = final_carry

        rho = Qarray.create(self.parameterize_density_matrix(final_params, dim))

        self._result = MLETomographyResult(
            rho=rho,
            params_history=history["params"],
            loss_history=history["loss"],
            grads_history=history["grads"],
            infidelity_history=history["infidelity"]
            if compute_infidelity_flag
            else None,
        )
        return self._result

    def quantum_state_tomography_direct(
        self,
    ) -> Qarray:

        """Perform quantum state tomography using direct inversion.
    
        This method reconstructs the quantum state from measurement results by 
        directly solving a system of linear equations. The method assumes that
        the measurement basis is complete and the measurement results are 
        noise-free.
    
        Returns:
            Qarray: Reconstructed quantum state.
        """

    # Compute overlaps of measurement and complete operator bases
        A = jnp.einsum("ijk,ljk->il", self.complete_basis, self.measurement_basis)
        # Solve the linear system to find the coefficients
        coefficients = jnp.linalg.solve(A, self.measurement_results)
        # Reconstruct the density matrix
        rho = jnp.einsum("i, ijk->jk", coefficients, self.complete_basis)

        return Qarray.create(rho)

    def plot_results(self):
        if self._result is None:
            raise ValueError(
                "No results to plot. Run quantum_state_tomography_mle first."
            )

        fig, ax = plt.subplots(1, figsize=(5, 4))
        if self._result.infidelity_history is not None:
            ax2 = ax.twinx()

        ax.plot(self._result.loss_history, color="C0")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("$\\mathcal{L}$", color="C0")
        ax.set_yscale("log")

        if self._result.infidelity_history is not None:
            ax2.plot(self._result.infidelity_history, color="C1")
            ax2.set_yscale("log")
            ax2.set_ylabel("$1-\\mathcal{F}$", color="C1")
            plt.grid(False)

        plt.show()

def tensor_basis(single_basis: Qarray, n: int) -> Qarray:
    """Construct n-fold tensor product basis from a single-system basis.

    Args:
        single_basis: The single-system operator basis as a Qarray.
        n: Number of tensor copies to construct.

    Returns:
        Qarray containing the n-fold tensor product basis operators.
        The resulting basis has b^n elements where b is the number
        of operators in the single-system basis.
    """

    dims = single_basis.dims

    single_basis = single_basis.data
    b, d, _ = single_basis.shape
    indices = jnp.stack(jnp.meshgrid(*[jnp.arange(b)] * n, indexing="ij"),
                        axis=-1).reshape(-1, n)  # shape (b^n, n)

    # Select the operators based on indices: shape (b^n, n, d, d)
    selected = single_basis[indices]  # shape: (b^n, n, d, d)

    # Vectorized Kronecker products
    full_basis = vmap(lambda ops: reduce(jnp.kron, ops))(selected)

    new_dims = tuple(tuple(x**n for x in row) for row in dims)

    return Qarray.create(full_basis, dims=new_dims, bdims=(b**n,))


def _quantum_process_tomography(
    map: Callable[[Qarray], Qarray],
    physical_state_basis: Qarray,
    physical_operator_basis: Qarray,
    logical_state_basis: Optional[Qarray] = None,
    logical_operator_basis: Optional[Qarray] = None,
) -> Qarray:

    if logical_state_basis is None:
        logical_state_basis = physical_state_basis
    if logical_operator_basis is None:
        logical_operator_basis = physical_operator_basis

    dsqr = logical_state_basis.bdims[-1]

    d = int(jnp.sqrt(dsqr+1))

    choi = Qarray.create(jnp.zeros((d, d))) ^ Qarray.create(jnp.zeros((d, d)))

    with (tqdm(total=d * d) as pbar):
        for k in range(dsqr):
                rho_k = physical_state_basis[k] @ physical_state_basis[k].dag()

                E_rho_k = map(rho_k)

                measurement_results = jnp.real(
                    jnp.einsum('ijk,jk->i', physical_operator_basis.data,
                               E_rho_k.data))

                QST = QuantumStateTomography(rho_guess=identity(d)/d,
                                             measurement_results=measurement_results,
                                             measurement_basis=logical_operator_basis,
                                             )
                res = QST.quantum_state_tomography_mle()
                r = res.rho
                print("----------------------")
                print(logical_state_basis[k] @ logical_state_basis[k].dag())
                print(r)
                choi += (logical_state_basis[k] @ logical_state_basis[k].dag()
                         ) ^ r
                pbar.update(1)
    return choi
