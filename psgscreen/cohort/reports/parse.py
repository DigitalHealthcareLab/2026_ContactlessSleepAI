"""Fields used by the study from one PSG report (blocks from extract.py).

Tables are read by row label and, for the respiratory table, by merged header names (two or three header rows),
so the same code reads all report templates of the three centres. Cells holding "-" or nothing are missing.

  hospital_id, date, age (from the date of birth when not given), sex, bmi (from height and weight when
  plausible, else the reported BMI), tib_min, tst_min,
  ahi       the A/H index stated in the impression ("A/H index: -" = 0), else the AHI row of the respiratory
            table (>= 3 % desaturation rule where present), else a bare "AHI <value>" in the impression
  rbd_status, rbd_evidence (rbd.py), rbdsq (score in the narrative part of the impression),
  study_type from the file name and text: MSLT, MWT, CPAP/BPAP titration, split-night, expanded EEG, diagnostic
"""
import os
import re
import unicodedata
from datetime import date

from psgscreen.cohort.reports.rbd import classify_rbd, rbdsq

DASH = {"-", "–", "—", "", "N/A", "NA", ".", "·"}
_FIELD_CODE = re.compile(r'DDE(?:AUTO)?\s+Tw\s+Ex\s*"[^"]*"(?:\s*\\#\s*"[^"]*")?', re.I)   # unevaluated Word fields
_FORMULA = re.compile(r'=\s*\(\s*[\d\s./*+-]+\)\s*\*?\s*\d*')
_MISC_FIELD = re.compile(r'FORMDROPDOWN|FORMTEXT|EMBED\s+\S+(?:\s+\\\S+)?|MERGEFORMAT', re.I)


def norm(s):
    s = re.sub(r'<[^>]{0,40}>', '', s or '')
    s = s.replace('오류! 문서 변수가 지원되지 않습니다.', '')      # Word "document variable not supported" residue
    if 'DDE' in s or '=(' in s or '= (' in s or 'FORM' in s or 'EMBED' in s:
        s = _MISC_FIELD.sub(' ', _FIELD_CODE.sub(' ', s))
        s = _FORMULA.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def num(s):
    """'85.6 %', ' / 110', '11.2/hr' -> float; dashes and blanks -> None."""
    if s is None:
        return None
    t = norm(str(s))
    if t in DASH:
        return None
    m = re.search(r'-?\d+(?:\.\d+)?', re.sub(r'^[/\s]+', '', t.replace(',', '')))
    return float(m.group(0)) if m else None


def cell(row, i):
    return num(row[i]) if (row is not None and i is not None and i < len(row)) else None


class Table(list):
    __slots__ = ("flat",)


def tables_of(blocks):
    out = []
    for b in blocks:
        if b["t"] == "tbl":
            t = Table(b["x"])
            t.flat = " || ".join(norm(c) for row in b["x"] for c in row)
            out.append(t)
    return out


def find_table(tables, pat):
    rx = re.compile(pat, re.I)
    return next((t for t in tables if rx.search(t.flat)), None)


def row_by(t, pat):
    """First row whose label (first cell) matches pat."""
    if t is None:
        return None
    rx = re.compile(pat, re.I)
    return next((row for row in t if row and rx.search(norm(row[0]))), None)


def col_map(t, nheader):
    """{header name: column} from the first nheader rows (first occurrence wins)."""
    m = {}
    for row in (t[:nheader] if t is not None else []):
        for i, c in enumerate(row):
            k = norm(c).lower().rstrip(':')
            if k and k not in DASH and k not in m:
                m[k] = i
    return m


def squash(s):
    return re.sub(r"\s+", " ", s or "").strip()


def block_text(b):
    return squash(b["x"]) if b["t"] == "p" else " || ".join(squash(c) for row in b["x"] for c in row)


IMP_START = re.compile(r"Clinical\s+Polysomnographer.{0,3}s\s+Diagnostic\s+Impression|^Diagnostic\s+Finding"
                       r"|^Conclusion\s*:?$|^Conclusion\s*:", re.I)
IMP_END = re.compile(r"^(Number\s+of\s+Respiration\s+Events|Respiratory\s+Event\s+Durations"
                     r"|Table\s+of\s+Respiratory\s+Events|Position(?:al)?\s+Summary|Arousal\s+Summary"
                     r"|Limb\s+Movements?\s+Summary|Diagnostic\s+SaO2|Heart\s+Rate\s+Analysis|Recommendation"
                     r"|Table\s+of\s+REM\s+Movement|Hypnogram)", re.I)


def impression_region(blocks):
    """Text from the impression / finding / conclusion header up to the next report section."""
    para = [i for i, b in enumerate(blocks) if b["t"] == "p"]
    s = next((i for i in para if IMP_START.search(squash(blocks[i]["x"]))), None)
    if s is None:
        return ""
    e = next((j for j in para if j > s and IMP_END.search(squash(blocks[j]["x"]))), len(blocks))
    return squash(" ".join(block_text(b) for b in blocks[s:e]))


HEADER_LABEL = re.compile(r'^(Patient|DOB|Age|Sex|Height|Weight|BMI|Date|Hospital ID|PSG Tech|Scored by Physician|'
                          r'Chief Complaints|Referring Department|Referring Physician|Medical history)\s*:?\s*$', re.I)


def parse_header(tables):
    out, t = {}, find_table(tables, r'Hospital\s*ID')
    for row in (t or []):
        cells = [norm(c) for c in row]
        for i, c in enumerate(cells):
            m = HEADER_LABEL.match(c)
            if m and i + 1 < len(cells) and cells[i + 1]:
                out.setdefault(m.group(1).lower().replace(' ', '_'), cells[i + 1])
    return out


def parse_date(s):
    for rx, order in ((r'(\d{1,2})/(\d{1,2})/(\d{4})', 'mdy'), (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ymd'),
                      (r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})', 'ymd')):
        m = re.search(rx, s or "")
        if m:
            a, b, c = m.groups()
            y, mo, d = (c, a, b) if order == 'mdy' else (a, b, c)
            try:
                return date(int(y), int(mo), int(d)).isoformat()
            except ValueError:
                return None
    return None


def _legacy_columns(row):
    """Old respiratory template (NREM / REM each split by position): column index by row length."""
    n = len(row)
    if n >= 12:
        return dict(nrem=1, rem=4, supine=8, nonsupine=10, total=11)
    if n == 10:
        return dict(nrem=1, rem=4, supine=7, nonsupine=8, total=9)
    if n == 9:
        return dict(nrem=1, rem=4, supine=7, nonsupine=None, total=8)
    return dict(nrem=1 if n > 1 else None, rem=4 if n > 4 else None, supine=None, nonsupine=None,
                total=n - 1 if n > 1 else None)


def table_ahi(tables):
    """Total AHI of the respiratory table."""
    t = find_table(tables, r'(Total\s+Apnea\s+Index|Apnea\s+Index\s*:|A/H\s*INDEX)')
    if t is None:
        return None
    if len(t) >= 3 and [norm(c).lower() for c in t[2]].count('supine') >= 2 and \
            'all' in [norm(c).lower() for c in t[2]]:
        r_ahi = row_by(t, r'^A\s*/\s*H\s*INDEX') or row_by(t, r'^A/H\s*Index')
        if r_ahi is not None:
            return cell(r_ahi, _legacy_columns(r_ahi)['total'])
        r_cnt = row_by(t, r'^Apnea\s*/\s*Hypopnea')
        if r_cnt is None:
            return None
        r_st = row_by(t, r'^Sleep\s+Time\s*\(min')       # index = 60 x events / sleep minutes
        st = [cell(r_st, _legacy_columns(r_st)[k]) for k in ('nrem', 'rem')] if r_st is not None else [None, None]
        tot, tst = cell(r_cnt, _legacy_columns(r_cnt)['total']), (st[0] or 0) + (st[1] or 0)
        return round(60.0 * tot / tst, 1) if (tot is not None and tst) else None
    r_ahi = (row_by(t, r'AHI\s*\(?[≥>]?\s*3\s*%') or row_by(t, r'A/H\s*Index\s*\(?[≥>]?\s*3\s*%')
             or row_by(t, r'^AHI\b') or row_by(t, r'^A/H\s*Index'))
    return cell(r_ahi, col_map(t, 3).get('total'))


def impression_ahi(imp):
    """(A/H index value or 0 for "A/H index: -", bare "AHI <value>")."""
    m = re.search(r'A\s*/\s*H\s*index\s*[:：]\s*(\d+(?:\.\d+)?)', imp, re.I)
    ah = float(m.group(1)) if m else (0.0 if re.search(r'A\s*/\s*H\s*index\s*[:：]\s*-', imp, re.I) else None)
    m = re.search(r'\bAHI\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(?:and|,|$|\n)', imp, re.I)
    return ah, (float(m.group(1)) if m else None)


STUDY = [('MSLT', r'\bMSLT\b'), ('MWT', r'\bMWT\b'), ('CPAP/BPAP titration', r'\b(CPAP|BPAP|APAP|PAP)\b'),
         ('split-night', r'\bsplit\b'), ('expanded EEG', r'\bexp\b')]


def study_type(path, text):
    base = unicodedata.normalize('NFC', os.path.basename(path))
    tags = [name for name, rx in STUDY if re.search(rx, base, re.I)]
    if re.search(r'Clinical\s+Polysomnographer.{0,3}s\s+CPAP\s+titration', text, re.I) and \
            'CPAP/BPAP titration' not in tags:
        tags.append('CPAP titration section')
    return ';'.join(tags) if tags else 'diagnostic'


def parse_report(path, blocks):
    tables = tables_of(blocks)
    imp = impression_region(blocks)
    h = parse_header(tables)
    exam_date, dob = parse_date(h.get('date')), parse_date(h.get('dob'))
    age = num(h.get('age'))
    if age is None and dob and exam_date:
        d0, d1 = date.fromisoformat(dob), date.fromisoformat(exam_date)
        age = d1.year - d0.year - ((d1.month, d1.day) < (d0.month, d0.day))

    ht, wt, bmi_rep = num(h.get('height')), num(h.get('weight')), num(h.get('bmi'))
    ht = ht if ht is not None and 50 <= ht <= 220 else None
    wt = wt if wt is not None and 5 <= wt <= 300 else None
    bmi = None
    if ht and wt:
        cand = round(wt / (ht / 100.0) ** 2, 2)
        bmi = cand if 10 <= cand <= 70 else None
    if bmi is None and bmi_rep is not None and 10 <= bmi_rep <= 70:
        bmi = bmi_rep

    sex = (h.get('sex') or '').strip()
    sex = 'M' if re.match(r'^m', sex, re.I) else ('F' if re.match(r'^f', sex, re.I) else None)
    m = re.match(r'^\s*(\d{5,})\s*[,\s]\s*(.*)$', h.get('patient', ''))
    hid = h.get('hospital_id') or (m.group(1) if m else None)
    hid = (re.sub(r'\D', '', hid) or None) if hid else None
    hid = hid.lstrip('0') if hid else None
    if not hid:                                             # fall back to the number before '~' in the file name
        m = re.match(r'^\s*0*(\d+)\s*~', os.path.basename(path))
        hid = m.group(1) if m else None

    ah, bare = impression_ahi(imp)
    ahi = ah if ah is not None else table_ahi(tables)
    ahi = ahi if ahi is not None else bare
    t = find_table(tables, r'Total\s+Sleep\s+Time\s*\(TST\)')
    status, evidence = classify_rbd(imp)
    return dict(hospital_id=hid, date=exam_date, age=age, sex=sex, bmi=bmi,
                tib_min=cell(row_by(t, r'^Time\s+In\s+Bed'), 1), tst_min=cell(row_by(t, r'^Total\s+Sleep\s+Time'), 1),
                ahi=ahi, rbd_status=status, rbd_evidence=evidence, rbdsq=rbdsq(imp[:3000]),
                study_type=study_type(path, squash(" ".join(block_text(b) for b in blocks))))
