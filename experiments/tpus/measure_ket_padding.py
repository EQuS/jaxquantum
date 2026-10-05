"""Measure the TPU lane-padding overhead of (N, 1) vs (N,) ket storage.

Motivation
----------
On a TPU, an array's minor (last) axis is padded up to the 128-element lane
width and the second-minor axis up to the 8-element sublane width. A ket stored
as an ``(N, 1)`` column therefore has its size-1 minor axis padded to 128 — the
physical buffer becomes ~128x larger than the N complex numbers it holds. This
script *measures* that, so the jaxquantum migration to 1-D ``(N,)`` ket storage
can be verified on real hardware.

What it does
------------
For a few sizes it compiles a trivial op on:
  * a legacy column ket  ``(N, 1)``           (the OLD storage)
  * a 1-D ket            ``(N,)``             (the NEW storage)
  * a batched legacy ket ``(B, N, 1)``
  * a batched 1-D ket    ``(B, N)``
and reports the compiled buffer size from ``compiled.memory_analysis()`` plus
the HLO layout from ``compiled.as_text()``. It prints an old-vs-new ratio and a
PASS/FAIL verdict.

On CPU/GPU there is no lane padding, so the ratio is ~1 (the script says so).
On TPU the legacy ``(N, 1)`` buffers should be ~128x the ``(N,)`` ones.

Run
---
    conda run -n jqt-env python experiments/tpus/measure_ket_padding.py

(Run on a TPU host to see the real blow-up; runs anywhere as a smoke test.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxquantum as jqt  # noqa: F401  (ensures x64 + same import path as users)


def _output_bytes(x) -> int:
    """Physical output-buffer size (bytes) of an identity copy of ``x``.

    ``memory_analysis().output_size_in_bytes`` reflects the *padded* device
    buffer, so it captures TPU lane padding that ``x.nbytes`` (logical) does not.
    """
    f = jax.jit(lambda a: a + 0)
    compiled = f.lower(x).compile()
    ma = compiled.memory_analysis()
    return int(ma.output_size_in_bytes)


def _hlo_layout_line(x) -> str:
    """A short snippet of the compiled HLO showing the buffer shape/layout."""
    f = jax.jit(lambda a: a + 0)
    text = f.lower(x).compile().as_text()
    for line in text.splitlines():
        if "ROOT" in line or "parameter(0)" in line:
            return line.strip()
    return text.splitlines()[0].strip() if text else "<no hlo>"


def main() -> None:
    platform = jax.devices()[0].platform
    print(f"JAX backend platform: {platform}")
    if platform != "tpu":
        print(
            "NOTE: not running on a TPU — CPU/GPU have no lane padding, so the "
            "old/new ratio below will be ~1. Run on a TPU host to see the "
            "(N,1) -> (N,128) blow-up.\n"
        )

    sizes = [(128, None), (1024, None), (4096, None), (1024, 8)]  # (N, batch)
    print(f"{'shape (old)':>14} {'shape (new)':>12} "
          f"{'old bytes':>12} {'new bytes':>12} {'ratio':>7}  verdict")
    print("-" * 72)

    any_blowup = False
    for N, B in sizes:
        if B is None:
            old = jnp.ones((N, 1), dtype=jnp.complex128)
            new = jnp.ones((N,), dtype=jnp.complex128)
            old_shape, new_shape = f"({N}, 1)", f"({N},)"
        else:
            old = jnp.ones((B, N, 1), dtype=jnp.complex128)
            new = jnp.ones((B, N), dtype=jnp.complex128)
            old_shape, new_shape = f"({B}, {N}, 1)", f"({B}, {N})"

        old_b = _output_bytes(old)
        new_b = _output_bytes(new)
        ratio = old_b / max(new_b, 1)
        # On TPU we expect a large ratio for the size-1 minor axis. Treat >4x as
        # "padding observed"; ~1x means no padding (CPU/GPU) which is also fine.
        if ratio > 4:
            any_blowup = True
            verdict = "PADDED (old wastes memory)"
        else:
            verdict = "no padding"
        print(f"{old_shape:>14} {new_shape:>12} {old_b:>12,} {new_b:>12,} "
              f"{ratio:>6.1f}x  {verdict}")

    print("\nHLO layout (old (N,1) vs new (N,)) for N=4096:")
    print("  old:", _hlo_layout_line(jnp.ones((4096, 1), dtype=jnp.complex128)))
    print("  new:", _hlo_layout_line(jnp.ones((4096,), dtype=jnp.complex128)))

    # End-to-end check: jaxquantum kets are now 1-D, and the sesolve carry stays
    # 1-D (no (N,1) reintroduced). Confirm the carried state buffer has no padded
    # minor axis.
    N = 1024
    H = jqt.create(N) @ jqt.destroy(N)
    psi = jqt.basis(N, 1)
    print(f"\njqt.basis({N},1).data.shape = {psi.data.shape}  (expected ({N},))")

    def rhs(psi_data):
        return -1j * jnp.einsum("ij,j->i", H.data, psi_data)

    carry_line = _hlo_layout_line(psi.data)
    print("sesolve RHS carry buffer (should be 1-D [N], not [N,1]/[N,128]):")
    print("  ", carry_line)
    _ = jax.jit(rhs).lower(psi.data).compile()  # smoke: compiles cleanly

    print("\nSummary:")
    if platform == "tpu":
        if any_blowup:
            print("  PASS: legacy (N,1) storage is lane-padded ~128x on this TPU; "
                  "the new (N,) storage avoids it.")
        else:
            print("  WARNING: expected padding on TPU but ratio was ~1 — "
                  "inspect memory_analysis()/as_text() manually.")
    else:
        print("  OK (CPU/GPU): no lane padding here. The migration's benefit is "
              "TPU-specific; re-run this on a TPU host to quantify it.")


if __name__ == "__main__":
    main()
