from nlp.verb_graph import VerbGraph


def test_preferred_punct():
    g = VerbGraph()
    g.add_sentence('run!')
    g.add_sentence('run!')
    g.add_sentence('run?')
    assert g.edges['run']['!'] == 2
    assert g.edges['run']['?'] == 1
    assert g.preferred_punct('run') == '!'
