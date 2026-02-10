import pyfuga
import pyfuga._fuga as _fuga


def test_get_luts_wrapper_delegates_to__fuga(monkeypatch):
    sentinel = object()

    def fake_get_luts(*args, **kwargs):
        # return something distinctive so we can assert delegation
        return (sentinel, args, kwargs)

    monkeypatch.setattr(_fuga, "get_luts", fake_get_luts)

    out = pyfuga.get_luts(1, 2, foo="bar")

    assert out[0] is sentinel
    assert out[1] == (1, 2)
    assert out[2] == {"foo": "bar"}
