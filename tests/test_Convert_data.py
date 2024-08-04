from pathlib import Path

import pandas as pd

from reliability.Convert_data import (
    FNRN_to_FR,
    FNRN_to_XCN,
    FR_to_FNRN,
    FR_to_XCN,
    XCN_to_FNRN,
    XCN_to_FR,
    xlsx_to_FNRN,
    xlsx_to_FR,
    xlsx_to_XCN,
)


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


def test_xlsx_to_XCN() -> None:
    X: list[int] = [13, 45, 78, 89, 102, 105]
    C: list[str] = ["F", "F", "F", "C", "C", "C"]
    N: list[int] = [1, 1, 1, 2, 2, 1]
    data = {"event time": X, "censor code": C, "number of events": N}
    XCN_df = pd.DataFrame(data=data, columns=["event time", "censor code", "number of events"])
    current_directory: Path = Path.cwd()
    file_name: Path = current_directory / "tests/_excel_files/XCN.xlsx"
    XCN_df.to_excel(file_name, index=False)
    XCN = xlsx_to_XCN(file_name)
    assert (X == XCN.X).all()
    assert (C == XCN.C).all()
    assert (N == XCN.N).all()
    file_name.unlink()


def test_xlsx_to_FR() -> None:
    failures: list[int | str] = [1, 1, 2, 2, 3]
    right_censored: list[int | str] = [9, 9, 9, 9, 8, 8, 7]
    f: list[int | str] = failures.copy()
    rc: list[int | str] = right_censored.copy()
    len_f: int = len(f)
    len_rc: int = len(rc)
    max_len: int = max(len_f, len_rc)
    if max_len != len_f:
        f.extend([""] * (max_len - len_f))
    if max_len != len_rc:
        rc.extend([""] * (max_len - len_rc))
    data = {"failures": f, "right censored": rc}
    FR_df = pd.DataFrame(data, columns=["failures", "right censored"])
    current_directory: Path = Path.cwd()
    file_name: Path = current_directory / "tests/_excel_files/FR.xlsx"
    FR_df.to_excel(file_name, index=False)
    FR = xlsx_to_FR(file_name)
    assert (failures == FR.failures).all()
    assert (right_censored == FR.right_censored).all()
    file_name.unlink()


def test_xlsx_to_FNRN():
    failures = [1, 2, 3]
    num_failures = [2, 2, 1]
    right_censored = [9, 8, 7]
    num_right_censored = [3, 2, 1]
    FR = FNRN_to_FR(
        failures=failures,
        num_failures=num_failures,
        right_censored=right_censored,
        num_right_censored=num_right_censored,
    )
    FNRN = FR_to_FNRN(
        failures=FR.failures,
        right_censored=FR.right_censored,
    )
    current_directory: Path = Path.cwd()
    file_name: Path = current_directory / "tests/_excel_files/FNRN.xlsx"
    FNRN._FR_to_FNRN__df.to_excel(file_name, index=False)
    FNRN_file = xlsx_to_FNRN(file_name)
    assert (FNRN.failures == FNRN_file.failures).all()
    assert (FNRN.right_censored == FNRN_file.right_censored).all()
    assert (FNRN.num_failures == FNRN_file.num_failures).all()
    assert (FNRN.num_right_censored == FNRN_file.num_right_censored).all()
    file_name.unlink()
