from effrag.compression import get_compressor
from effrag.data import Doc

DOCS = [
    Doc("Eiffel Tower", "The Eiffel Tower is 330 metres tall.", True, True),
    Doc("Paris", "Paris is the capital of France."),
]


def test_baselines():
    full = get_compressor("full").compress("How tall is the Eiffel Tower?", DOCS)
    assert full.ratio == 1.0 and "Document [2]" in full.context
    oracle = get_compressor("oracle").compress("q", DOCS)
    assert "Paris is" not in oracle.context
    assert get_compressor("closedbook").compress("q", DOCS).comp_tokens == 0
    assert get_compressor("truncate", budget=5).compress("q", DOCS).comp_tokens <= 6
