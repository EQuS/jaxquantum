""" Helpers for ATS coupler tutorials. """

import qcsys as qs 
import jax.numpy as jnp
import jaxquantum as jqt

N_CONS = {
    "resonator_a": {
        "bare": 8,
        "normal": 8,
        "truncated": 8,
    },
    "ats": {
        "bare": 5,
        "normal": 100,
        "truncated": 10,
    },
    "resonator_b": {
        "bare": 8,
        "normal": 8,
        "truncated": 8,
    }
}

DRIVE_STRENGTH = 0.49

def create_devices_linear_rar(params):

    f_a = params.get("ResonatorA_frequency", 5)
    f_b = params.get("ResonatorB_frequency", 7)

    _, Ec_a, El_a = qs.calculate_lambda_over_four_resonator_zpf(f_a, 50)

    resonator_a = qs.Resonator.create(
        N_CONS["resonator_a"]["bare"],
        {"Ec": Ec_a, "El": El_a},
        N_pre_diag=N_CONS["resonator_a"]["bare"],
        use_linear=True,
        label="A"
    )

    _, Ec_b, El_b = qs.calculate_lambda_over_four_resonator_zpf(f_b, 50)

    resonator_b = qs.Resonator.create(
        N_CONS["resonator_b"]["bare"],
        {"Ec": Ec_b, "El": El_b},
        N_pre_diag=N_CONS["resonator_b"]["bare"],
        use_linear=True,
        label="B"
    )

    Ec_c = params.get("ATS__E_C", 0.0726)
    El_c = params.get("ATS__E_L", 62.40)
    Ej_c = params.get("ATS__E_J", 37.0)
    dEj_c = params.get("ATS__dE_J", 0.0)
    Ej2_c = params.get("ATS__E_J_2", 0.0)
    phi_sum_c = params.get("ATS__phi_sum_ext", 0.25)
    phi_delta_c = params.get("ATS__phi_delta_ext", 0.25)

    ats = qs.ATS.create(
        N_CONS["ats"]["bare"],
        {
            "Ec": Ec_c, 
            "El": El_c, 
            "Ej": Ej_c,
            "dEj": dEj_c,
            "Ej2": Ej2_c,
            "phi_sum_ext": phi_sum_c,
            "phi_delta_ext": phi_delta_c,
        },
        N_pre_diag=N_CONS["ats"]["bare"],
        use_linear=True,
        label=""
    )

    return resonator_a, ats, resonator_b

def create_system_linear_rar(params):
    resonator_a, ats, resonator_b = create_devices_linear_rar(params)

    J_ac = params.get("ResonatorA_ATS__J", 0.1)
    J_cb = params.get("ATS_ResonatorB__J", 0.1)

    g_ac = J_ac * resonator_a.n_zpf() * ats.n_zpf()
    g_cb = J_cb * ats.n_zpf() * resonator_b.n_zpf()

    devices = [resonator_a, ats, resonator_b]
    Ns = [device.N for device in devices]

    a = qs.promote(resonator_a.ops["a"], 0, Ns)
    a_dag = qs.promote(resonator_a.ops["a_dag"], 0, Ns)

    c = qs.promote(ats.ops["a"], 1, Ns)
    c_dag = qs.promote(ats.ops["a_dag"], 1, Ns)

    b = qs.promote(resonator_b.ops["a"], 2, Ns)
    b_dag = qs.promote(resonator_b.ops["a_dag"], 2, Ns)

    couplings = []
    couplings.append(-g_ac * (a - a_dag) @ (c - c_dag))
    couplings.append(-g_cb * (c - c_dag) @ (b - b_dag))

    system = qs.System.create(devices, couplings=couplings)
    system.params["g_ac"] = g_ac
    system.params["g_cb"] = g_cb

    return system


def get_metrics_linear_rar(params):
    system = create_system_linear_rar(params)
    resonator_a = system.devices[0]
    ats = system.devices[1]
    resonator_b = system.devices[2]

    """diagonalize"""
    Es, kets = system.calculate_eig()

    """extract eigenvectors"""
    vac = kets[0, 0, 0]
    name_to_device = {device.label: device for device in system.devices}

    e = {}
    e[system.devices[0].label] = kets[1, 0, 0]
    e[system.devices[1].label] = kets[0, 1, 0]
    e[system.devices[2].label] = kets[0, 0, 1]
    e.keys()

    """extract participations"""
    ϕ = {device.label: {} for device in system.devices}

    for j, d1 in enumerate(system.devices):
        phi_0 = system.promote(d1.ops["phi"], j).data
        for d2 in system.devices:
            ϕ[d1.label][d2.label] = jqt.dag_data(e[d2.label]) @ phi_0 @ vac

    metrics = {}

    metrics[f"ω_{system.devices[0].label}"] = Es[1, 0, 0] - Es[0, 0, 0]
    metrics[f"ω_{system.devices[1].label}"] = Es[0, 1, 0] - Es[0, 0, 0]
    metrics[f"ω_{system.devices[2].label}"] = Es[0, 0, 1] - Es[0, 0, 0]

    drive_strength = params.get("ATS__drive_strength", DRIVE_STRENGTH)
    metrics[f"g_ex"] = ϕ["ATS"]["ResonatorA"] * ϕ["ATS"]["ResonatorB"] * drive_strength * ats.params["Ej"]
    metrics[f"g_3"] = (1/2) * ϕ["ATS"]["ResonatorA"]**2 * ϕ["ATS"]["ResonatorB"] * drive_strength * ats.params["Ej"]
    metrics[f"g_cd"] = ϕ["ATS"]["ResonatorA"]**2 * ϕ["ATS"]["ResonatorB"] * drive_strength * ats.params["Ej"]
    return ϕ, metrics, system



def get_devices_normal_rar(params):
    """set up devices"""
    ϕ0, metrics0, system0 = get_metrics_linear_rar(params)

    # units are GHz and ns
    ω_a = metrics0["ω_ResonatorA"]
    ω_c = metrics0["ω_ATS"]
    ω_b = metrics0["ω_ResonatorB"]

    ϕa_zpf = ϕ0["ResonatorA"]["ResonatorA"]
    ϕc_zpf = ϕ0["ATS"]["ATS"]
    ϕb_zpf = ϕ0["ResonatorB"]["ResonatorB"]

    # Ratios
    Ec_over_El_a = ϕa_zpf**4 / 2
    Ec_over_El_c = ϕc_zpf**4 / 2
    Ec_over_El_b = ϕb_zpf**4 / 2

    Ec_a = jnp.sqrt(ω_a**2 / 8 * Ec_over_El_a)
    Ec_c = jnp.sqrt(ω_c**2 / 8 * Ec_over_El_c)
    Ec_b = jnp.sqrt(ω_b**2 / 8 * Ec_over_El_b)

    resonator_a = qs.Resonator.create(
        N_CONS["resonator_a"]["truncated"],
        {"Ec": Ec_a, "El": Ec_a / Ec_over_El_a},
        N_pre_diag=N_CONS["resonator_a"]["normal"],
    )

    Ej_c = system0.devices[1].params["Ej"]
    dEj_c = system0.devices[1].params["dEj"]
    Ej2_c = system0.devices[1].params["Ej2"]


    resonator_a = qs.Resonator.create(
        N_CONS["resonator_a"]["truncated"],
        {"Ec": Ec_a, "El": Ec_a / Ec_over_El_a},
        N_pre_diag=N_CONS["resonator_a"]["normal"],
        label="A"
    )

    ats = qs.ATS.create(
        N_CONS["ats"]["truncated"],
        {
            "Ec": Ec_c,
            "El": Ec_c / Ec_over_El_c,
            "Ej": Ej_c,
            "dEj": dEj_c,
            "Ej2": Ej2_c,
            "phi_sum_ext": system0.devices[1].params["phi_sum_ext"],
            "phi_delta_ext": system0.devices[1].params["phi_delta_ext"],
        },
        N_pre_diag=N_CONS["ats"]["normal"],
        use_linear=False, # NOTE: here we use the full non-linear hamiltonian for the ATS, this helps with quantum number assignment
        label=""
    )

    resonator_b = qs.Resonator.create(
        N_CONS["resonator_b"]["truncated"],
        {"Ec": Ec_b, "El": Ec_b / Ec_over_El_b},
        N_pre_diag=N_CONS["resonator_b"]["normal"],
        label="B"
    )

    return resonator_a, ats, resonator_b, ϕ0, metrics0, system0 



def get_system_normal_rar(params):
    """set up devices"""
    resonator_a, ats, resonator_b, ϕ0, metrics0, system0  = get_devices_normal_rar(params)

    """add couplings"""
    devices = [resonator_a, ats, resonator_b]
    a_indx = 0
    c_indx = 1
    b_indx = 2
    Ns = [device.N for device in devices]


    a = qs.promote(resonator_a.ops["a"], a_indx, Ns)
    a_dag = qs.promote(resonator_a.ops["a_dag"], a_indx, Ns)
    phi_a = ϕ0["ATS"]["ResonatorA"] * (a + a_dag)

    b = qs.promote(resonator_b.ops["a"], b_indx, Ns)
    b_dag = qs.promote(resonator_b.ops["a_dag"], b_indx, Ns)
    phi_b = ϕ0["ATS"]["ResonatorB"] * (b + b_dag)

    c = qs.promote(ats.ops["a"], c_indx, Ns)
    c_dag = qs.promote(ats.ops["a_dag"], c_indx, Ns)
    phi_c = qs.promote(ats.ops["phi"], c_indx, Ns)

    # sanity check
    phi_c_alternative = ϕ0["ATS"]["ATS"] * (c + c_dag)
    # assert jnp.abs(phi_c- phi_c_alternative).max() < 1e-10

    phi = phi_a + phi_c + phi_b
    id_op_c = qs.promote(ats.ops["id"], c_indx, Ns)

    couplings = []
    coupling_term = 0 
    coupling_term += ats.get_H_nonlinear(phi)
    coupling_term -= ats.get_H_nonlinear(phi_c)
    couplings.append(coupling_term)

    system = qs.System.create(devices, couplings=couplings)
    system.params["phi_c"] = phi_c
    system.params["phi_c_alternative"] = phi_c_alternative
    system.params["phi_a"] = phi_a
    system.params["phi_b"] = phi_b
    system.params["phi"] = phi
    system.params["a"] = a 
    system.params["b"] = b
    system.params["c"] = c
    return system, ϕ0, metrics0, system0


def get_metrics_normal_rar(params):
    """set up devices"""
    system, ϕ0, metrics0, system0 = get_system_normal_rar(params)
    a_indx = 2
    c_indx = 1
    b_indx = 0

    epsilon_p = params.get("ATS__drive_strength", DRIVE_STRENGTH)

    resonator_a = system.devices[a_indx]
    ats = system.devices[c_indx]
    resonator_b = system.devices[b_indx]

    Es, kets = system.calculate_eig()

    n_ats = 0
    K_a = (Es[2:, n_ats, 0] - Es[1:-1, n_ats, 0]) - (
        Es[1:-1, n_ats, 0] - Es[0:-2, n_ats, 0]
    )
    K_b = (Es[0, n_ats, 2:] - Es[0, n_ats, 1:-1]) - (
        Es[0, n_ats, 1:-1] - Es[0, n_ats, 0:-2]
    )

    metrics = {}
    metrics["K_a"] = K_a
    metrics["K_b"] = K_b
    metrics["E"] = Es

    return metrics, system, ϕ0, metrics0, system0