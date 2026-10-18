from effrag.metrics import best_subspan_em, exact_match, f1


def test_best_subspan_em():
    assert best_subspan_em("It was Wilhelm Conrad Röntgen.", ["Wilhelm Conrad Röntgen"]) == 1.0
    assert best_subspan_em("Marie Curie", ["Wilhelm Conrad Röntgen"]) == 0.0


def test_exact_match_and_f1():
    assert exact_match("The Beatles", ["beatles"]) == 1.0
    assert 0 < f1("the band Beatles", ["The Beatles"]) < 1
