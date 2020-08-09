"""Development tasks."""

import os
from pathlib import Path
from shutil import which

import nox

os.environ.update({'PDM_IGNORE_SAVED_PYTHON': '1', 'PDM_USE_VENV': '1'})

PYTHON_VERSION_MIN, *_ = PYTHON_VERSIONS = ['3.10', '3.11']
FILES = ['src', 'tests', 'docs', 'noxfile.py']


def check_cli(session: nox.Session, args: list[str]) -> None:
    """Check the CLI arguments.

    Arguments:
        session: The nox session.
        args: The available CLI arguments.
    """
    if len(session.posargs) > 1 or session.posargs[0] not in args:
        available_args = ', '.join([f'`{arg}`' for arg in args])
        session.skip(
            f'Available options should be one of {available_args}. Instead `{" ".join(session.posargs)}` was given',
        )


@nox.session(python=PYTHON_VERSION_MIN)
def docs(session: nox.Session) -> None:
    """Build, serve or deploy the documentation.

    Arguments:
        session: The nox session.
    """
    check_cli(session, ['serve', 'build', 'deploy'])
    session.run('pdm', 'install', '-dG', 'docs', external=True)
    if session.posargs[0] != 'deploy':
        session.run('mkdocs', session.posargs[0])
    else:
        if which('vercel') is None:
            session.skip('You should first install the `vercel` command')
        session.run(
            'vercel',
            'pull',
            '--yes',
            '--environment=production',
            '--token=${{ secrets.VERCEL_TOKEN }}',
            external=True,
        )
        session.run('vercel', 'deploy', '--prod', '--token=${{ secrets.VERCEL_TOKEN }}', 'site', external=True)


@nox.session(python=PYTHON_VERSION_MIN)
@nox.parametrize('file', FILES)
def formatting(session: nox.Session, file: str) -> None:
    """Format the code.

    Arguments:
        session: The nox session.
        file: The file to be formatted.
    """
    check_cli(session, ['all', 'code', 'docstrings'])
    session.run('pdm', 'install', '-dG', 'formatting', '--no-default', external=True)
    if session.posargs[0] in ['code', 'all']:
        session.run('black', file)
    if session.posargs[0] in ['docstrings', 'all']:
        session.run('docformatter', '--in-place', '--recursive', '--close-quotes-on-newline', file)


@nox.session(python=PYTHON_VERSION_MIN)
@nox.parametrize('file', FILES)
def checks(session: nox.Session, file: str) -> None:
    """Check code quality, dependencies or type annotations.

    Arguments:
        session: The nox session.
        file: The file to be checked.
    """
    check_cli(session, ['all', 'quality', 'dependencies', 'types'])
    session.run('pdm', 'install', '-dG', 'checks', '--no-default', external=True)
    if session.posargs[0] in ['quality', 'all']:
        session.run('ruff', file)
    if session.posargs[0] in ['types', 'all']:
        session.run('mypy', file)
    if session.posargs[0] in ['dependencies', 'all']:
        requirements_path = (Path(session.create_tmp()) / 'requirements.txt').as_posix()
        session.run('pdm', 'export', '-f', 'requirements', '--without-hashes', '-o', requirements_path, external=True)
        session.run('safety', 'check', '-r', requirements_path)


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run tests and coverage.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'tests', external=True)
    env = {'COVERAGE_FILE': f'.coverage.{session.python}'}
    if session.posargs:
        session.run('pytest', '-n', 'auto', '-k', *session.posargs, 'tests', env=env)
    else:
        session.run('pytest', '-n', 'auto', 'tests', env=env)
    session.run('coverage', 'combine')
    session.run('coverage', 'report')
    session.run('coverage', 'html')


@nox.session(python=PYTHON_VERSION_MIN)
def changelog(session: nox.Session) -> None:
    """Add news fragment or build changelog.

    Arguments:
        session: The nox session.
    """
    check_cli(session, ['add', 'build'])
    session.run('pdm', 'install', '-dG', 'changelog', external=True)
    import click

    if session.posargs[0] == 'add':
        issue_num = click.prompt('Issue number (start with + for orphan fragment)')
        frag_type = click.prompt('News fragment type', type=str)
        session.run('towncrier', 'create', '--edit', f'{issue_num}.{frag_type}.txt')
    if session.posargs[0] == 'build':
        version = click.prompt('Incremented version number to use')
        session.run('towncrier', 'build', '--yes', '--version', version)


@nox.session(python=PYTHON_VERSION_MIN)
def release(session: nox.Session) -> None:
    """Kick off a release process.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'release', external=True)
    import click

    version = click.prompt('Incremented version number to use')
    session.run('git', 'checkout', 'master', external=True)
    try:
        session.run('git', 'tag', '-a', version, external=True)
        session.run('git', 'push', '--tags', external=True)
    finally:
        session.run('pdm', 'build', external=True)
        session.run('twine', 'upload', '--skip-existing', 'dist/*')
