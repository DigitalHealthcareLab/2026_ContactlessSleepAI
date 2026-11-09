"""PSG report files -> blocks: [{"t": "p", "x": paragraph text}, {"t": "tbl", "x": [[cell, ...], ...]}].

The format is decided by the file's magic bytes, not its extension (many ".doc" reports are OOXML):
  'PK'        OOXML (.docx): word/document.xml parsed directly so that table rows and cells are kept
  D0 CF 11 E0 Word 97 binary: text recovered from the piece table, cells / rows from the cell marks
"""
import zipfile
from xml.etree import ElementTree as ET

import olefile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _para_text(p):
    out = []
    for node in p.iter():
        if node.tag == W + "t":
            out.append(node.text or "")
        elif node.tag == W + "tab":
            out.append("\t")
        elif node.tag in (W + "br", W + "cr"):
            out.append("\n")
    return "".join(out)


def docx_blocks(path):
    with zipfile.ZipFile(path) as z:
        body = ET.fromstring(z.read("word/document.xml")).find(W + "body")
    blocks = []
    for child in (body if body is not None else []):
        if child.tag == W + "p":
            t = _para_text(child)
            if t.strip():
                blocks.append({"t": "p", "x": t})
        elif child.tag == W + "tbl":
            rows = [["\n".join(_para_text(p) for p in tc.findall(W + "p")) for tc in tr.findall(W + "tc")]
                    for tr in child.findall(W + "tr")]
            if rows:
                blocks.append({"t": "tbl", "x": rows})
    return blocks


def _u16(b, o):
    return int.from_bytes(b[o:o + 2], "little")


def _u32(b, o):
    return int.from_bytes(b[o:o + 4], "little")


def doc_text(path):
    """Text of a Word 97 document (control characters kept) from the piece table in the Clx."""
    ole = olefile.OleFileIO(path)
    try:
        wd = ole.openstream("WordDocument").read()
        table = "1Table" if (_u16(wd, 0x000A) >> 9) & 1 else "0Table"
        if not ole.exists(table):
            table = "0Table" if ole.exists("0Table") else "1Table"
        tb = ole.openstream(table).read()
    finally:
        ole.close()
    clx = tb[_u32(wd, 0x01A2):_u32(wd, 0x01A2) + _u32(wd, 0x01A6)]
    i, plc = 0, None
    while i < len(clx):                      # skip Prc entries (0x01) up to the Pcdt (0x02)
        if clx[i] == 0x01:
            i += 3 + _u16(clx, i + 1)
        elif clx[i] == 0x02:
            plc = clx[i + 5:i + 5 + _u32(clx, i + 1)]
            break
        else:
            i += 1
    if not plc:
        return ""
    n = (len(plc) - 4) // 12                 # n + 1 character positions (4 bytes) and n pieces (8 bytes)
    cps = [_u32(plc, 4 * k) for k in range(n + 1)]
    out = []
    for k in range(n):
        fc = _u32(plc[4 * (n + 1) + 8 * k:4 * (n + 1) + 8 * k + 8], 2)
        ncp = cps[k + 1] - cps[k]
        if fc & 0x40000000:                  # compressed piece: 8-bit cp1252
            off = (fc & 0x3FFFFFFF) >> 1
            out.append(wd[off:off + ncp].decode("cp1252", errors="replace"))
        else:
            off = fc & 0x3FFFFFFF
            out.append(wd[off:off + 2 * ncp].decode("utf-16-le", errors="replace"))
    return "".join(out)


_CTRL = {"\x0b": "\n", "\x0c": "\n", "\x1e": "-", "\x1f": "", "\x13": "", "\x14": "", "\x15": "",
         "\x01": "", "\x02": "", "\x08": "", "\x05": ""}


def _clean(s):
    for k, v in _CTRL.items():
        s = s.replace(k, v)
    return s.strip()


def doc_blocks(path):
    """Paragraphs end with \\r; table cells end with \\x07 and a second \\x07 closes the row."""
    txt = doc_text(path)
    blocks, rows, cells, buf = [], [], [], []

    def flush_table():
        nonlocal rows
        if rows:
            blocks.append({"t": "tbl", "x": rows})
            rows = []

    i, n = 0, len(txt)
    while i < n:
        ch = txt[i]
        if ch == "\x07":
            cells.append(_clean("".join(buf))); buf = []
            if i + 1 < n and txt[i + 1] == "\x07":
                rows.append(cells); cells = []
                i += 1
            i += 1
            continue
        if ch == "\r":
            t = _clean("".join(buf)); buf = []
            if cells:
                cells.append(t)
            else:
                if rows:
                    flush_table()
                if t:
                    blocks.append({"t": "p", "x": t})
            i += 1
            continue
        buf.append(ch)
        i += 1
    if cells:
        rows.append(cells)
    flush_table()
    t = _clean("".join(buf))
    if t:
        blocks.append({"t": "p", "x": t})
    return blocks


def read_blocks(path):
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] == b"PK":
        return docx_blocks(path)
    if magic == b"\xd0\xcf\x11\xe0":
        return doc_blocks(path)
    raise ValueError(f"unknown file format (magic {magic.hex()})")
