# Contributing

Contributions are welcome, and they are greatly appreciated.

## Environment setup

This project uses [PDM](https://github.com/pdm-project/pdm) to manage the various dependencies. To setup the development
environment you can follow the next steps:

- Install [PDM](https://github.com/pdm-project/pdm).

- Fork and clone the repository.

- `pdm install -dG :all` from the root of the project to install the main and development dependencies.

## Tasks

This project uses [nox](https://nox.thea.codes/en/stable/) to run tasks. Please check the `noxfile.py` at the root of the
project for more details. You can run any of the following commands that corresponds to a particular task:

- `pdm build-docs` to build the documentation.

- `pdm serve-docs` to serve the documentation.

- `pdm serve-docs` to deploy the documentation.

- `pdm format-code` to format the code.

- `pdm check-quality` to check the code quality.

- `pdm check-dependencies` to check for vulnerabilities in dependencies.

- `pdm check-types` to check type annotations.

- `pdm test` to run the tests.

- `pdm coverage` to run the tests coverage.

- `pdm release` to release a new Python package with an upadted version.

## Development

The next steps should be followed during development:

- `git checkout -b new-branch-name` to create a new branch and then modify the code.
- `pdm format-code` to auto-format the code.
- `pdm check-quality` to check code quality and fix any warnings.
- `pdm check-types` to check type annotations and fix any warnings.
- `pdm test` to run the tests.
- `pdm serve-docs` if you updated the documentation or the project dependencies to check that everything looks good.

## Pull Request guidelines

Link to any related issue in the Pull Request message. We also recommend using fixups:

```bash
git commit --fixup=SHA
```

Once all the changes are approved, you can squash your commits:

```bash
git rebase -i --autosquash master
```

And force-push:

```bash
git push -f
```
