from pyfuga import utils


def test_profile_decorator_behaves_like_decorator():
    calls = {}

    @utils.profile
    def add_one(x):
        calls["x"] = x
        return x + 1

    result = add_one(2)

    # Whatever profile is (real or dummy), the wrapped function should still:
    # - be callable
    # - return the original result
    # - execute the body
    assert result == 3
    assert calls["x"] == 2
