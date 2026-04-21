*This markdown document is not processed into a docs page.*

# Publishing to PyPI

Publishing is automated via the `.github/workflows/publish.yml` workflow using [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (no API tokens required).

Steps:
1. Update the `version` field in `pyproject.toml` and commit to `main`.
2. Create and push a tag matching the version:
   ```bash
   git tag v<VERSION>
   git push origin v<VERSION>
   ```
   For example: `v0.3.0`
3. The GitHub Actions workflow triggers automatically and publishes to PyPI (pending approval from @Phionx).
4. Once the workflow succeeds, go to [github.com/EQuS/jaxquantum/releases](https://github.com/EQuS/jaxquantum/releases), find the new tag, and create a Release with notes.

> **One-time setup:** Trusted Publishing must be configured on PyPI under jaxquantum → Settings → Trusted Publishers, and a `pypi` environment must exist in the GitHub repo settings. This has already been done for jaxquantum.
