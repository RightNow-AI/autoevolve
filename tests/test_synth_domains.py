import pytest

from autoevolve.synth.domains import classify_domain


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("Fuse this Triton kernel", "triton-kernel"),
        ("Reduce GPU memory traffic", "triton-kernel"),
        ("Make this Python function faster", "python-speedup"),
        ("Optimize Python parsing", "python-speedup"),
        ("Find a shorter TSP tour", "routing-heuristic"),
        ("Improve the delivery route", "routing-heuristic"),
        ("Fit an equation to these samples", "symbolic-regression"),
        ("Discover a compact formula", "symbolic-regression"),
        ("Improve this implementation", "general"),
    ],
)
def test_classify_domain_keyword_map(goal, expected):
    assert classify_domain(goal) == expected
