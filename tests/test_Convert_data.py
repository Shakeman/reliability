from reliability.Convert_data import FNRN_to_FR, FNRN_to_XCN, FR_to_FNRN, FR_to_XCN, XCN_to_FNRN, XCN_to_FR


def test_FR_to_FNRN():
    FNRN = FR_to_FNRN(
        failures=[8, 15, 15, 20, 25, 30, 30, 30, 30, 32, 32, 32],
        right_censored=[17, 17, 50, 50, 50, 50, 78, 78, 78, 78, 90],
    )
    assert (FNRN.failures == [8, 15, 20, 25, 30, 32]).all()
    assert (FNRN.num_failures == [1, 2, 1, 1, 4, 3]).all()
    assert (FNRN.right_censored == [17, 50, 78, 90]).all()
    assert (FNRN.num_right_censored == [2, 4, 4, 1]).all()


def test_FNRN_to_FR():
    FR = FNRN_to_FR(failures=[1, 2, 3], num_failures=[1, 1, 2], right_censored=[9, 8, 7], num_right_censored=[5, 4, 4])
    assert (FR.failures == [1, 2, 3, 3]).all()
    assert (
        FR.right_censored
        == [
            9,
            9,
            9,
            9,
            9,
            8,
            8,
            8,
            8,
            7,
            7,
            7,
            7,
        ]
    ).all()


def test_XCN_to_FR():
    FR = XCN_to_FR(
        X=[12, 15, 18, 32, 35, 38, 60],
        C=[1, 1, 1, 0, 0, 0, 0],
        quantities=[1, 1, 1, 2, 2, 1, 3],
        failure_code=1,
        censor_code=0,
    )
    assert (FR.failures == [12.0, 15.0, 18.0]).all()
    assert (FR.right_censored == [32.0, 32.0, 35.0, 35.0, 38.0, 60.0, 60.0, 60.0]).all()


def test_XCN_to_FNRN():
    FNRN = XCN_to_FNRN(X=[1, 2, 3, 7, 8, 9], C=["f", "f", "f", "c", "c", "c"], N=[1, 2, 2, 3, 2, 1])
    assert (FNRN.failures == [1.0, 2.0, 3.0]).all()
    assert (FNRN.num_failures == [1, 2, 2]).all()
    assert (FNRN.right_censored == [7.0, 8.0, 9.0]).all()
    assert (FNRN.num_right_censored == [3, 2, 1]).all()


def test_FR_to_XCN():
    XCN = FR_to_XCN(failures=[1, 1, 2, 2, 3], right_censored=[9, 9, 9, 9, 8, 8, 7])
    assert ([1, 2, 3, 7, 8, 9] == XCN.X).all()
    assert (["F", "F", "F", "C", "C", "C"] == XCN.C).all()
    assert ([2, 2, 1, 1, 2, 4] == XCN.N).all()


def test_FNRN_to_XCN():
    XCN = FNRN_to_XCN(
        failures=[1, 2, 3],
        num_failures=[2, 2, 1],
        right_censored=[9, 8, 7],
        num_right_censored=[3, 2, 1],
    )
    assert ([1, 2, 3, 7, 8, 9] == XCN.X).all()
    assert (["F", "F", "F", "C", "C", "C"] == XCN.C).all()
    assert ([2, 2, 1, 1, 2, 3] == XCN.N).all()
