""" Sweeping tools. """

from copy import deepcopy
import itertools
from tempfile import NamedTemporaryFile
import jax.numpy as jnp
from tqdm import tqdm
import os



def run_sweep(params, sweep_params, metrics_func, fixed_kwargs=None, data=None, is_parallel=False, save_file=None, data_save_mode="end", return_errors=False):
    """ Run a sweep over a single parameter, or multiple parameters.

    Args:
        params (dict): The base parameters to sweep over.
        sweep_params (dict): The parameters to sweep over.
            key: The parameter name.
            value: The list of values to sweep over.
        metrics_func (function): The function to evaluate the metrics.
        fixed_params (dict, optional): The fixed parameters to send into metrics_func. Defaults to None.
        data (dict, optional): The data to append to. Defaults to None.
        is_parallel (bool, optional): Whether to sweep through the sweep_params lists in parallel or through their cartesian product. Defaults to False.
        save_file (str, optional): The file to save the data to. Defaults to None, in which case data is saved to a temporary file, which will be deleted upon closing (e.g. during garbage collection).
        data_save_mode (str, optional): The mode to save the data. Defaults to None. 
            Options are: 
                "no" - don't save data 
                "end" - save data at the end of the sweep
                "during" - save data during and at the end of the sweep
    Returns:
        dict: The data after the sweep.
    """
    if data is None:
        data = {}
        run = -1
    else:
        run = max(data.keys())

    assert data_save_mode in ["no", "end", "during"], "Invalid data_save_mode."

    if data_save_mode in ["during", "end"]:
        if isinstance(save_file, str):
            print("Saving data to: ", save_file)
            dirname = os.path.dirname(save_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        else:
            save_file = save_file or NamedTemporaryFile()
            print("Saving data to a temporary file: ", save_file.name)
        

    fixed_kwargs = fixed_kwargs or {}

    if is_parallel:
        sweep_length = len(list(sweep_params.values())[0])
        assert [len(vals) == sweep_length for vals in sweep_params.values()], "Parallel sweep parameters must have the same length."

        errors = []
        try:
            for j in tqdm(range(sweep_length)):
                run += 1
                data[run] = {}
                data[run]["params"] = deepcopy(params)
                sweep_point_info = {
                    "labels": [],
                    "values": [],
                    "indices": [],
                }
                for key, vals in sweep_params.items():
                    data[run]["params"][key] = vals[j]
                    sweep_point_info["labels"].append(key)
                    sweep_point_info["values"].append(vals[j])
                    sweep_point_info["indices"].append(j)
                data[run]["results"] = metrics_func(data[run]["params"], **fixed_kwargs)
                data[run]["sweep_point_info"] = sweep_point_info
                if data_save_mode == "during":
                    jnp.savez(save_file, data=data, sweep_params=sweep_params, params=params)
        except Exception as e:
            errors.append(str(e))
            print("Error during run: ", errors[-1])
        
        try:
            if data_save_mode in ["during", "end"]:
                jnp.savez(save_file, data=data, sweep_params=sweep_params, params=params, error=None)
        except Exception as e:
            errors.append(str(e))
            print("Error during saving: ", errors[-1])

        if return_errors:
            return data, errors
        else:
            return data
    else:
        # Product Sweep
        sweep_points = list(itertools.product(*list(sweep_params.values())))
        sweep_points_indxs = list(itertools.product(*[list(range(len(vals))) for vals in list(sweep_params.values())]))
        sweep_point_labels = list(sweep_params.keys())

        errors = []

        try:
            with tqdm(total=len(sweep_points)) as pbar:
                for j, sweep_point in enumerate(sweep_points):
                    run += 1
                    data[run] = {}
                    data[run]["params"] = deepcopy(params)
                    sweep_point_info = {
                        "labels": [],
                        "values": [],
                        "indices": [],
                    }
                    for i, key in enumerate(sweep_point_labels):
                        data[run]["params"][key] = sweep_point[i]
                        sweep_point_info["labels"].append(key)
                        sweep_point_info["values"].append(sweep_point[i])
                        sweep_point_info["indices"].append(sweep_points_indxs[j][i])
                    data[run]["results"] = metrics_func(data[run]["params"], **fixed_kwargs)
                    data[run]["sweep_point_info"] = sweep_point_info
                    pbar.update(1)
                    if data_save_mode == "during":
                        jnp.savez(save_file, data=data, sweep_params=sweep_params, params=params)
        except Exception as e:
            errors.append(str(e))
            print("Error during run: ", errors[-1])
            
        try:
            if data_save_mode in ["during", "end"]:
                jnp.savez(save_file, data=data, sweep_params=sweep_params, params=params)
        except Exception as e:
            errors.append(str(e))
            print("Error during saving: ", errors[-1])

        if return_errors:
            return data, errors
        else:
            return data