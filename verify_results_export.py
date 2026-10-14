"""Verify row append/update on a disposable copy, preserving manual records."""
import copy
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import cfgs
import experiment_results as exporter


def cells(path):
    namespace = {"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(item.itertext()) for item in ET.fromstring(
                archive.read("xl/sharedStrings.xml")).findall("m:si",namespace)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        values = {}
        for cell in sheet.findall(".//m:sheetData/m:row/m:c",namespace):
            value = cell.find("m:v",namespace)
            if cell.get("t") == "s":
                result = strings[int(value.text)]
            elif cell.get("t") == "inlineStr":
                result = "".join(cell.find("m:is",namespace).itertext())
            elif cell.get("t") == "str":
                result = value.text
            elif value is not None:
                result = float(value.text)
            else:
                continue
            values[cell.get("r")] = result
        return values


def main():
    Path(".verification").mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="results-export-",dir=".verification")).resolve()
    original = exporter.WORKBOOK
    disposable = scratch / "test.xlsx"
    shutil.copy2(original,disposable)
    before = cells(disposable)
    request = exporter.build_request(cfgs)
    request["workbook"] = str(disposable)
    request["runId"] = "synthetic-append-check"
    request["checkpoint"] = "synthetic-checkpoint.pth"
    request["test"] = dict(dice=.8,iou=.7,hd95=4.5,precision=.85,recall=.9)
    request["validation"] = dict(dice=.82,iou=.72,hd95=3.5,precision=.86,recall=.91)
    exporter.WORKBOOK = disposable
    try:
        exporter.write_request(request)
        appended = cells(disposable)
        for address,value in before.items():
            assert appended[address] == value, address
        assert "B4" not in appended or not appended["B4"]
        assert appended["D4"] == .8 and appended["E4"] == .7
        assert appended["F4"] == 4.5 and appended["I4"] == .82
        exporter.write_request(copy.deepcopy(request))
        assert cells(disposable) == appended
        print("PASS: append, validation/test values, blank new model, manual values preserved, duplicate replay")
    finally:
        exporter.WORKBOOK = original
    print(f"Synthetic workbook retained at {disposable}")


if __name__ == "__main__":
    main()
