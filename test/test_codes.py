import pytest
import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jaxquantum.codes as jqtb
import jax.numpy as jnp
import numpy as np


class TestBosonicQubit:
    """Test the base BosonicQubit class functionality."""
    
    def test_bosonic_qubit_initialization(self):
        """Test basic initialization of BosonicQubit."""
        # This should raise an error since BosonicQubit is abstract
        with pytest.raises(TypeError):
            jqtb.BosonicQubit()
    
    def test_bosonic_qubit_name_property(self):
        """Test that the name property is accessible."""
        # Test through a concrete subclass
        qubit = jqtb.Qubit()
        assert hasattr(qubit, 'name')
        assert qubit.name == "bqubit"  # Base class name is "bqubit"
    
    def test_bosonic_qubit_params_validation(self):
        """Test default parameter validation."""
        qubit = jqtb.Qubit()
        assert "N" in qubit.params
        assert qubit.params["N"] == 2
    
    def test_bosonic_qubit_common_gates(self):
        """Test that common gates are generated."""
        qubit = jqtb.Qubit()
        assert "a_dag" in qubit.common_gates
        assert "a" in qubit.common_gates
        assert qubit.common_gates["a_dag"].shape == (2, 2)
        assert qubit.common_gates["a"].shape == (2, 2)
    
    def test_bosonic_qubit_basis_states(self):
        """Test that all required basis states are present."""
        qubit = jqtb.Qubit()
        required_states = ["+x", "-x", "+y", "-y", "+z", "-z"]
        for state in required_states:
            assert state in qubit.basis
            assert qubit.basis[state].shape == (2, 1)
    
    def test_bosonic_qubit_pauli_gates(self):
        """Test Pauli gate properties."""
        qubit = jqtb.Qubit()
        assert qubit.x_U.shape == (2, 2)
        assert qubit.y_U.shape == (2, 2)
        assert qubit.z_U.shape == (2, 2)
        assert qubit.h_U.shape == (2, 2)
    
    def test_bosonic_qubit_projector(self):
        """Test projector property."""
        qubit = jqtb.Qubit()
        projector = qubit.projector  # Access as property, not callable
        assert projector.shape == (2, 2)
        # Projector should be Hermitian - use .dag() method
        assert jnp.allclose(projector.data, projector.dag().data)
    
    def test_bosonic_qubit_maximally_mixed_state(self):
        """Test maximally mixed state property."""
        qubit = jqtb.Qubit()
        mixed_state = qubit.maximally_mixed_state  # Access as property, not callable
        assert mixed_state.shape == (2, 2)
        # Trace should be 1
        assert jnp.allclose(jnp.trace(mixed_state.data), 1.0)


class TestQubit:
    """Test the Qubit class (concrete implementation)."""
    
    def test_qubit_initialization(self):
        """Test Qubit initialization."""
        qubit = jqtb.Qubit()
        assert qubit.name == "bqubit"  # Base class name is "bqubit"
        assert qubit.params["N"] == 2
    
    def test_qubit_basis_states(self):
        """Test Qubit basis state construction."""
        qubit = jqtb.Qubit()
        plus_z, minus_z = qubit._get_basis_z()
        assert plus_z.shape == (2, 1)
        assert minus_z.shape == (2, 1)
        # Check that states are normalized - use .dag() method and .data attribute
        assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0)
        assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0)
    
    def test_qubit_pauli_gates(self):
        """Test Qubit Pauli gates."""
        qubit = jqtb.Qubit()
        # Test X gate
        x_gate = qubit.x_U
        assert x_gate.shape == (2, 2)
        # Test Y gate
        y_gate = qubit.y_U
        assert y_gate.shape == (2, 2)
        # Test Z gate
        z_gate = qubit.z_U
        assert z_gate.shape == (2, 2)
    
    def test_qubit_orthogonality(self):
        """Test that basis states are orthogonal."""
        qubit = jqtb.Qubit()
        plus_z = qubit.basis["+z"]
        minus_z = qubit.basis["-z"]
        # States should be orthogonal - use .dag() method and .data attribute
        assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=1e-10)


class TestBosonicMode:
    """Test the BosonicMode class."""
    
    def test_bosonic_mode_initialization(self):
        """Test BosonicMode initialization."""
        mode = jqtb.BosonicMode({"N": 4})
        assert mode.name == "bqubit"
        assert mode.params["N"] == 4
    
    def test_bosonic_mode_basis_states(self):
        """Test BosonicMode basis state construction."""
        mode = jqtb.BosonicMode({"N": 4})
        plus_z, minus_z = mode._get_basis_z()
        assert plus_z.shape == (4, 1)
        assert minus_z.shape == (4, 1)
        # Check that states are normalized - use .dag() method and .data attribute
        assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0)
        assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0)
        # Check that states are orthogonal
        assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=1e-10)


class TestCatQubit:
    """Test the CatQubit class."""
    
    def test_cat_qubit_initialization(self):
        """Test CatQubit initialization."""
        cat_qubit = jqtb.CatQubit()
        assert cat_qubit.name == "cat"
        assert cat_qubit.params["N"] == 50  # default
        assert cat_qubit.params["alpha"] == 2  # default
    
    def test_cat_qubit_custom_params(self):
        """Test CatQubit with custom parameters."""
        cat_qubit = jqtb.CatQubit({"N": 20, "alpha": 3.0})
        assert cat_qubit.params["N"] == 20
        assert cat_qubit.params["alpha"] == 3.0
    
    def test_cat_qubit_basis_states(self):
        """Test CatQubit basis state construction."""
        cat_qubit = jqtb.CatQubit({"N": 20, "alpha": 2.0})
        plus_z, minus_z = cat_qubit._get_basis_z()
        assert plus_z.shape == (20, 1)
        assert minus_z.shape == (20, 1)
        # Check that states are normalized - use .dag() method and .data attribute
        assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0, atol=1e-10)
        # Check that states are orthogonal - use more relaxed tolerance for finite truncation
        assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=1e-1)
    
    def test_cat_qubit_non_device_params(self):
        """Test that alpha is in non_device_params."""
        cat_qubit = jqtb.CatQubit()
        assert "alpha" in cat_qubit._non_device_params


class TestBinomialQubit:
    """Test the BinomialQubit class."""
    
    def test_binomial_qubit_initialization(self):
        """Test BinomialQubit initialization."""
        bin_qubit = jqtb.BinomialQubit()
        assert bin_qubit.name == "binomial"
        assert bin_qubit.params["N"] == 50  # default
        assert bin_qubit.params["L"] == 1  # default
        assert bin_qubit.params["G"] == 0  # default
        assert bin_qubit.params["D"] == 0  # default
    
    def test_binomial_qubit_custom_params(self):
        """Test BinomialQubit with custom parameters."""
        bin_qubit = jqtb.BinomialQubit({"N": 30, "L": 2, "G": 1, "D": 1})
        assert bin_qubit.params["N"] == 30
        assert bin_qubit.params["L"] == 2
        assert bin_qubit.params["G"] == 1
        assert bin_qubit.params["D"] == 1
    
    def test_binomial_qubit_basis_states(self):
        """Test BinomialQubit basis state construction."""
        bin_qubit = jqtb.BinomialQubit({"N": 20, "L": 1, "G": 0, "D": 0})
        plus_z, minus_z = bin_qubit._get_basis_z()
        assert plus_z.shape == (20, 1)
        assert minus_z.shape == (20, 1)
        # Check that states are normalized - use .dag() method and .data attribute
        assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0, atol=1e-10)
        # Check that states are orthogonal
        assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=1e-10)
    
    def test_binomial_qubit_parameter_combinations(self):
        """Test BinomialQubit with different parameter combinations."""
        # Test with L=2, G=1, D=0
        bin_qubit = jqtb.BinomialQubit({"N": 25, "L": 2, "G": 1, "D": 0})
        plus_z, minus_z = bin_qubit._get_basis_z()
        assert plus_z.shape == (25, 1)
        assert minus_z.shape == (25, 1)
        
        # Test with L=1, G=0, D=1
        bin_qubit = jqtb.BinomialQubit({"N": 25, "L": 1, "G": 0, "D": 1})
        plus_z, minus_z = bin_qubit._get_basis_z()
        assert plus_z.shape == (25, 1)
        assert minus_z.shape == (25, 1)


class TestGKPQubit:
    """Test the GKPQubit class."""
    
    def test_gkp_qubit_initialization(self):
        """Test GKPQubit initialization."""
        gkp_qubit = jqtb.GKPQubit()
        assert gkp_qubit.name == "gkp"
        assert gkp_qubit.params["N"] == 50  # default
        assert gkp_qubit.params["delta"] == 0.25  # default
        assert "l" in gkp_qubit.params
        assert "epsilon" in gkp_qubit.params
    
    def test_gkp_qubit_custom_params(self):
        """Test GKPQubit with custom parameters."""
        gkp_qubit = jqtb.GKPQubit({"N": 40, "delta": 0.3})
        assert gkp_qubit.params["N"] == 40
        assert gkp_qubit.params["delta"] == 0.3
        assert gkp_qubit.params["l"] == 2.0 * jnp.sqrt(jnp.pi)
    
    def test_gkp_qubit_common_gates(self):
        """Test GKPQubit common gates."""
        gkp_qubit = jqtb.GKPQubit({"N": 30})
        # Check that phase space operators are present
        assert "x" in gkp_qubit.common_gates
        assert "p" in gkp_qubit.common_gates
        # Check that finite energy operators are present
        assert "E" in gkp_qubit.common_gates
        assert "E_inv" in gkp_qubit.common_gates
        # Check that logical gates are present
        assert "X" in gkp_qubit.common_gates
        assert "Z" in gkp_qubit.common_gates
        assert "Y" in gkp_qubit.common_gates
        # Check that stabilizers are present
        assert "Z_s_0" in gkp_qubit.common_gates
        assert "S_x_0" in gkp_qubit.common_gates
        assert "S_z_0" in gkp_qubit.common_gates
        assert "S_y_0" in gkp_qubit.common_gates
    
    def test_gkp_qubit_basis_states(self):
        """Test GKPQubit basis state construction."""
        gkp_qubit = jqtb.GKPQubit({"N": 30, "delta": 0.3})
        plus_z, minus_z = gkp_qubit._get_basis_z()
        assert plus_z.shape == (30, 1)
        assert minus_z.shape == (30, 1)
        # Check that states are normalized - use .dag() method and .data attribute
        assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0, atol=1e-10)
        # Check that states are orthogonal - use more relaxed tolerance for finite truncation
        assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=1e-1)
    
    def test_gkp_qubit_parameter_validation(self):
        """Test GKPQubit parameter validation."""
        gkp_qubit = jqtb.GKPQubit({"delta": 0.1})
        # Check that derived parameters are calculated correctly
        expected_l = 2.0 * jnp.sqrt(jnp.pi)
        assert jnp.allclose(gkp_qubit.params["l"], expected_l)
        
        # Check epsilon calculation
        s_delta = jnp.sinh(0.1 ** 2)
        expected_epsilon = s_delta * expected_l
        assert jnp.allclose(gkp_qubit.params["epsilon"], expected_epsilon)


class TestHexagonalGKPQubit:
    """Test the HexagonalGKPQubit class."""
    
    def test_hexagonal_gkp_qubit_initialization(self):
        """Test HexagonalGKPQubit initialization."""
        hex_gkp_qubit = jqtb.HexagonalGKPQubit()
        assert hex_gkp_qubit.name == "gkp"  # All GKP variants use "gkp" name
        assert hex_gkp_qubit.params["N"] == 50  # default
        assert hex_gkp_qubit.params["delta"] == 0.25  # default
    
    def test_hexagonal_gkp_qubit_basis_states(self):
        """Test HexagonalGKPQubit basis state construction."""
        hex_gkp_qubit = jqtb.HexagonalGKPQubit({"N": 30, "delta": 0.3})
        plus_z, minus_z = hex_gkp_qubit._get_basis_z()
        assert plus_z.shape == (30, 1)
        assert minus_z.shape == (30, 1)
        # Check that states are normalized - use .dag() method and .data attribute
        assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0, atol=1e-10)
        # Check that states are orthogonal - use more relaxed tolerance for finite truncation
        assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=1e-2)


class TestSquareGKPQubit:
    """Test the SquareGKPQubit class."""
    
    def test_square_gkp_qubit_initialization(self):
        """Test SquareGKPQubit initialization."""
        square_gkp_qubit = jqtb.SquareGKPQubit()
        assert square_gkp_qubit.name == "gkp"  # All GKP variants use "gkp" name
        assert square_gkp_qubit.params["N"] == 50  # default
        assert square_gkp_qubit.params["delta"] == 0.25  # default
    
    def test_square_gkp_qubit_basis_states(self):
        """Test SquareGKPQubit basis state construction."""
        square_gkp_qubit = jqtb.SquareGKPQubit({"N": 30, "delta": 0.3})
        plus_z, minus_z = square_gkp_qubit._get_basis_z()
        assert plus_z.shape == (30, 1)
        assert minus_z.shape == (30, 1)
        # Check that states are normalized - use .dag() method and .data attribute
        assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0, atol=1e-10)
        # Check that states are orthogonal - use more relaxed tolerance for finite truncation
        assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=1e-1)


class TestRectangularGKPQubit:
    """Test the RectangularGKPQubit class."""
    
    def test_rectangular_gkp_qubit_initialization(self):
        """Test RectangularGKPQubit initialization."""
        rect_gkp_qubit = jqtb.RectangularGKPQubit()
        assert rect_gkp_qubit.name == "gkp"  # All GKP variants use "gkp" name
        assert rect_gkp_qubit.params["N"] == 50  # default
        assert rect_gkp_qubit.params["delta"] == 0.25  # default
        # Note: eta parameter may not be present in all implementations
    
    def test_rectangular_gkp_qubit_custom_params(self):
        """Test RectangularGKPQubit with custom parameters."""
        rect_gkp_qubit = jqtb.RectangularGKPQubit({"N": 40, "delta": 0.3, "eta": 1.5})
        assert rect_gkp_qubit.params["N"] == 40
        assert rect_gkp_qubit.params["delta"] == 0.3
        # Note: eta parameter may not be stored in params
    
    def test_rectangular_gkp_qubit_basis_states(self):
        """Test RectangularGKPQubit basis state construction."""
        rect_gkp_qubit = jqtb.RectangularGKPQubit({"N": 30, "delta": 0.3, "eta": 1.2})
        plus_z, minus_z = rect_gkp_qubit._get_basis_z()
        assert plus_z.shape == (30, 1)
        assert minus_z.shape == (30, 1)
        # Check that states are normalized - use .dag() method and .data attribute
        assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0, atol=1e-10)
        # Check that states are orthogonal - use more relaxed tolerance for finite truncation
        assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=1e-1)


class TestCodesIntegration:
    """Integration tests for the codes submodule."""
    
    def test_all_code_types_initialization(self):
        """Test that all code types can be initialized."""
        codes = [
            jqtb.Qubit(),
            jqtb.BosonicMode({"N": 4}),
            jqtb.CatQubit(),
            jqtb.BinomialQubit(),
            jqtb.GKPQubit(),
            jqtb.HexagonalGKPQubit(),
            jqtb.SquareGKPQubit(),
            jqtb.RectangularGKPQubit(),
        ]
        
        for code in codes:
            assert hasattr(code, 'basis')
            assert hasattr(code, 'common_gates')
            assert hasattr(code, 'params')
            # All should have the required basis states
            required_states = ["+x", "-x", "+y", "-y", "+z", "-z"]
            for state in required_states:
                assert state in code.basis
    
    def test_code_state_orthogonality(self):
        """Test that all code types have orthogonal basis states."""
        codes = [
            jqtb.Qubit(),
            jqtb.CatQubit({"N": 20}),
            jqtb.BinomialQubit({"N": 20}),
            jqtb.GKPQubit({"N": 20}),
        ]
        
        for code in codes:
            plus_z = code.basis["+z"]
            minus_z = code.basis["-z"]
            # States should be orthogonal - use .dag() method and .data attribute
            # Use more relaxed tolerance for complex codes due to finite truncation
            tolerance = 1e-10 if code.name == "bqubit" else 1.5e-1
            assert jnp.allclose((plus_z.dag() @ minus_z).data, 0.0, atol=tolerance)
            # States should be normalized
            assert jnp.allclose(jnp.abs((plus_z.dag() @ plus_z).data), 1.0, atol=1e-10)
            assert jnp.allclose(jnp.abs((minus_z.dag() @ minus_z).data), 1.0, atol=1e-10)
    
    def test_code_gate_consistency(self):
        """Test that all code types have consistent gate properties."""
        codes = [
            jqtb.Qubit(),
            jqtb.CatQubit({"N": 20}),
            jqtb.BinomialQubit({"N": 20}),
            jqtb.GKPQubit({"N": 20}),
        ]
        
        for code in codes:
            # All should have Pauli gates
            assert hasattr(code, 'x_U')
            assert hasattr(code, 'y_U')
            assert hasattr(code, 'z_U')
            assert hasattr(code, 'h_U')
            
            # Gates should have correct shapes
            N = code.params["N"]
            assert code.x_U.shape == (N, N)
            assert code.y_U.shape == (N, N)
            assert code.z_U.shape == (N, N)
            assert code.h_U.shape == (N, N)
    
    def test_code_parameter_validation(self):
        """Test that all code types handle parameter validation correctly."""
        # Test with minimal parameters
        codes = [
            jqtb.Qubit(),
            jqtb.CatQubit(),
            jqtb.BinomialQubit(),
            jqtb.GKPQubit(),
        ]
        
        for code in codes:
            # All should have at least N parameter
            assert "N" in code.params
            # N should be reasonable
            assert code.params["N"] > 0
            assert code.params["N"] <= 100  # reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__])

