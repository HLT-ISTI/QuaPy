import pytest

def test_import():
    import quapy as qp
    assert qp.__version__ is not None
