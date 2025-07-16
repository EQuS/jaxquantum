
This is a non-exhaustive, but growing list of common pitfalls when using `jaxquantum`.

# JAX

!!! quote 
    When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has [*“una anima di pura programmazione funzionale”*](https://www.sscardapane.it/iaml-backup/jax-intro/). ~ JAX docs

Often the sharp bits at poking you originate from the paradigm within which JAX operates. The JAX developers have compiled a great list of these: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html 


# Operations

## Tensor product shorthand `^`

Note that the shorthand for tensor product `^` is evaluated after other basic math operations (e.g. `+,-,*,/,@`). So, when using this shorthand, it is best practice to use parentheses. 


!!! example "Tensor shorthand."
    For example, the following code will fail:
    ```python
    jqt.identity(2)^jqt.identity(3) + jqt.identity(2)^jqt.identity(3)
    ```

    !!! failure "Output"
        ```text
        ValueError: Dimensions are incompatible: ((3,), (3,)) and ((2,), (2,))
        ```
    This is because `jqt.identity(3) + jqt.identity(2)` is running before the tensor products.


    Instead, we should use parentheses to specify the order of operations to begin with `^`.

    ```python
    (jqt.identity(2)^jqt.identity(3)) + (jqt.identity(2)^jqt.identity(3))
    ```

    !!! success "Output"
        ```text
        Quantum array: dims = ((2, 3), (2, 3)), bdims = (), shape = (6, 6), type = oper
        Qarray data =
        [[2.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 2.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 2.+0.j 0.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 0.+0.j 2.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 0.+0.j 0.+0.j 2.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 2.+0.j]]
        ```

