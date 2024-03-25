from pathlib import Path

from reliability.Convert_data import FR_to_FNRN

FNRN = FR_to_FNRN(
    failures=[8, 15, 15, 20, 25, 30, 30, 30, 30, 32, 32, 32],
    right_censored=[17, 17, 50, 50, 50, 50, 78, 78, 78, 78, 90],
)
FNRN.print()
test_path: Path = Path.cwd().parent / "tests"
excel_path = test_path / "excel files"
print(test_path)
