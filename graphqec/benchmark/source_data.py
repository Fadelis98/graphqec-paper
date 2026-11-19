"""Deterministic stdlib-only XLSX writer/reader for the Source Data workbook.

Floats are stored with ``repr`` so every value round-trips exactly; the reader understands the
files written here (inline strings, numbers, booleans) and needs no spreadsheet package.
"""

import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_INVALID_SHEET_CHARS = set('[]:*?/\\')


def _col(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _clean_text(text: str) -> str:
    return "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)


def _cell(ref: str, value, header: bool = False) -> str:
    style = ' s="1"' if header else ""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}" t="b"{style}><v>{int(bool(value))}</v></c>'
    if isinstance(value, (int, np.integer)):
        return f'<c r="{ref}"{style}><v>{int(value)}</v></c>'
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return ""
        if math.isinf(value):
            value = "inf" if value > 0 else "-inf"
        else:
            return f'<c r="{ref}"{style}><v>{repr(value)}</v></c>'
    text = escape(_clean_text(str(value)))
    return f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">{text}</t></is></c>'


def frame_rows(frame) -> list[list]:
    """DataFrame -> [header] + rows with native Python scalars (NaN -> None)."""
    rows = [list(map(str, frame.columns))]
    for record in frame.itertuples(index=False, name=None):
        out = []
        for value in record:
            if value is None:
                out.append(None)
            elif isinstance(value, (float, np.floating)) and math.isnan(float(value)):
                out.append(None)
            elif isinstance(value, (np.bool_, bool)):
                out.append(bool(value))
            elif isinstance(value, np.integer):
                out.append(int(value))
            elif isinstance(value, np.floating):
                out.append(float(value))
            else:
                try:
                    import pandas as pd
                    if value is pd.NA:
                        out.append(None)
                        continue
                except ImportError:  # pragma: no cover
                    pass
                out.append(value)
        rows.append(out)
    return rows


def write_xlsx(path: Path, sheets: list[tuple[str, list[list]]]) -> None:
    names = [name for name, _ in sheets]
    for name in names:
        if len(name) > 31 or set(name) & _INVALID_SHEET_CHARS:
            raise ValueError(f"invalid sheet name {name!r}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate sheet names")
    files: dict[str, str] = {}
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, len(sheets) + 1))
    files["[Content_Types].xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{overrides}</Types>')
    files["_rels/.rels"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    sheet_tags = "".join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
                         for i, name in enumerate(names, start=1))
    files["xl/workbook.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{_NS}" xmlns:r="{_NS_R}"><sheets>{sheet_tags}</sheets></workbook>')
    rels = "".join(
        f'<Relationship Id="rId{i}" Type="{_NS_R}/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(sheets) + 1))
    rels += f'<Relationship Id="rId{len(sheets) + 1}" Type="{_NS_R}/styles" Target="styles.xml"/>'
    files["xl/_rels/workbook.xml.rels"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}'
        '</Relationships>')
    files["xl/styles.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<styleSheet xmlns="{_NS}">'
        '<fonts count="2"><font><sz val="10"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><name val="Arial"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>')
    for i, (_, rows) in enumerate(sheets, start=1):
        body = []
        for r_index, row in enumerate(rows):
            cells = "".join(_cell(f"{_col(c)}{r_index + 1}", value, header=(r_index == 0))
                            for c, value in enumerate(row))
            body.append(f'<row r="{r_index + 1}">{cells}</row>')
        files[f"xl/worksheets/sheet{i}.xml"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<worksheet xmlns="{_NS}" xmlns:r="{_NS_R}"><sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>'
            f'<sheetData>{"".join(body)}</sheetData></worksheet>')
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files, key=lambda n: (n != "[Content_Types].xml", n)):
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name].encode("utf-8"))


def read_xlsx(path: Path) -> dict[str, list[list]]:
    """Minimal reader for the files written above (used by the audit and the tests)."""
    import xml.etree.ElementTree as ET

    ns = {"m": _NS, "r": _NS_R}
    out: dict[str, list[list]] = {}
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        names = [s.attrib["name"] for s in workbook.find("m:sheets", ns)]
        for i, name in enumerate(names, start=1):
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{i}.xml"))
            rows = []
            for row in root.find("m:sheetData", ns):
                values: dict[int, object] = {}
                for cell in row:
                    ref = cell.attrib["r"]
                    letters = "".join(ch for ch in ref if ch.isalpha())
                    col = 0
                    for ch in letters:
                        col = col * 26 + (ord(ch) - 64)
                    kind = cell.attrib.get("t")
                    if kind == "inlineStr":
                        values[col - 1] = cell.find("m:is/m:t", ns).text or ""
                    elif kind == "b":
                        values[col - 1] = cell.find("m:v", ns).text == "1"
                    else:
                        text = cell.find("m:v", ns).text
                        values[col - 1] = float(text) if any(c in text for c in ".eE") else int(text)
                width = max(values) + 1 if values else 0
                rows.append([values.get(c) for c in range(width)])
            out[name] = rows
    return out
