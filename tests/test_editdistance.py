
def test_run():
    import editdistance
    assert editdistance.eval('aaa', 'aaba') == 1
