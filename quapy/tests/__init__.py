"""
Fast unit tests live in ``test_*.py`` and are intended to run without network
access or large external resources.

Slow integration tests that download datasets or depend on optional stacks live
in ``integration_*.py`` and are meant to be run explicitly, e.g.:

    python -m unittest quapy.tests.integration_datasets
    python -m unittest quapy.tests.integration_methods
"""
