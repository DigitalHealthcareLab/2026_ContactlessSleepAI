"""RBD status from the diagnostic impression / conclusion of a PSG report.

Questionnaire and scale mentions (RBDSQ, RBDSS) are masked first. The text is split into clauses (sentence ends,
semicolons, but / however / whereas / although); each clause that mentions RBD is classified, in this order:
indeterminate (cannot be evaluated) > negative > pseudo-RBD only > probable (possibility, should be considered,
clinical correlation, ...) > definite (suggests / compatible with / diagnosed as ...; probable if the clause
relies on a questionnaire or history). A mention matching no rule counts as probable.
Report status: definite > probable > pseudo_only > indeterminate > negative > none.
"""
import re

QUESTIONNAIRE = re.compile(
    r'RBD[\s-]*(?:screening\s*)?(?:questionnaire|Quesionnaire|SQ|Q)\b\s*(?:was\s*)?[:：;]?\s*'
    r'(?:not\s+done|not\s+checked|N/?A|\d+)?', re.I)
RBDSQ_LONG = re.compile(
    r'REM\s*sleep\s*behaviou?r\s*disorder\s*screening\s*questionnaire\s*[:：]?\s*\d*', re.I)
RBDSS = re.compile(r'RBDSS[^)\n]*\)?', re.I)
RBD_TOKEN = re.compile(r'REM\s*sleep\s*behaviou?r\s*disorder|REM\s*behaviou?r\s*disorder|\bRBD\b', re.I)
PSEUDO = re.compile(r'pseudo[\s-]*(?:RBD|REM\s*sleep\s*behaviou?r\s*disorder)', re.I)
RBD_ANY = r'(?:REM\s*sleep\s*behaviou?r\s*disorder|REM\s*behaviou?r\s*disorder|RBD)'
MOD = (r'(?:probable\s+|possible\s+|clinical\s+|early\s+|prodromal\s+|preclinical\s+|subclinical\s+|'
       r'suspicious\s+|definite\s+|concurrent\s+|typical\s+)*')

NEGATIVE = [
    re.compile(r'\bno\b[^.;]{0,90}?' + RBD_ANY, re.I),
    re.compile(r'\bnone\s+of\s+[^.;]{0,60}?' + RBD_ANY, re.I),
    re.compile(RBD_ANY + r'[^.;]{0,120}?(?:were|was|are|is)\s+not\s+'
               r'(?:recorded|found|observed|noted|seen|detected|demonstrated|shown)', re.I),
    re.compile(r'\bnot\s+(?:recorded|found|observed|noted|seen|detected|demonstrated|shown|suggestive)'
               r'[^.;]{0,80}?' + RBD_ANY, re.I),
    re.compile(r'(?:rule|ruled)\s+out\s+[^.;]{0,40}?' + RBD_ANY, re.I),
    re.compile(RBD_ANY + r'[^.;]{0,60}?\b(?:unlikely|excluded)\b', re.I),
    re.compile(r'\bdifferent\s+from\s+' + RBD_ANY, re.I),
    re.compile(r'lack\s+of\s+evidence[^.;]{0,70}?' + RBD_ANY, re.I),
    re.compile(r'(?:does|do|did)\s+not\s+support[^.;]{0,50}?' + RBD_ANY, re.I),
    re.compile(r'insufficient\s+evidence[^.;]{0,70}?' + RBD_ANY, re.I),
    re.compile(RBD_ANY + r'[^.;]{0,60}?(?:is|was|were|are)\s+not\s+supported', re.I),
]
INDETERMINATE = [
    re.compile(RBD_ANY + r'[^.;]{0,100}?(?:cannot|could\s+not|can\s+not|was\s+impossible|'
               r'uncheckable|unevaluable)[^.;]{0,60}?', re.I),
    re.compile(r'(?:cannot|could\s+not|can\s+not|unable\s+to)\s+[^.;]{0,60}?'
               r'(?:define|determine|evaluate|assess|check)[^.;]{0,60}?' + RBD_ANY, re.I),
    re.compile(r'(?:assessment|evaluation|presence)\s+of\s+' + RBD_ANY +
               r'[^.;]{0,60}?(?:impossible|not\s+possible)', re.I),
]
POSSIBLE = [
    re.compile(r'possibilit\w+\s+(?:for|of)\s+[^.;]{0,90}?' + RBD_ANY, re.I),
    re.compile(RBD_ANY + r'[^.;]{0,80}?(?:should\s+be\s+considered|'
               r'cannot\s+be\s+(?:excluded|ruled\s+out))', re.I),
    re.compile(r'(?:required|needed|recommended|could\s+be\s+required)\s+to\s+diagnos\w+'
               r'[^.;]{0,40}?' + RBD_ANY, re.I),
    re.compile(r'(?:clinical\s+correlation|consideration)[^.;]{0,80}?' + RBD_ANY, re.I),
    re.compile(r'suggest\w*\s+(?:the\s+)?(?:possibilit\w+|suspicious)\s+(?:of\s+)?' + MOD + RBD_ANY, re.I),
    re.compile(RBD_ANY + r'[\s-]*like\b', re.I),
]
POSITIVE = [
    re.compile(r'suggest\w*\s+(?:the\s+)?(?:presence\s+of\s+)?' + MOD + r'[^.;]{0,25}?' + RBD_ANY, re.I),
    re.compile(r'suspect\w*\s+(?:the\s+)?(?:presence\s+of\s+)?' + MOD + RBD_ANY, re.I),
    re.compile(r'(?:compatible|consistent)\s+with\s+' + MOD + r'[^.;]{0,30}?' + RBD_ANY, re.I),
    re.compile(r'\b(?:diagnosed|diagnosis)\s+(?:as|of|with)\s+' + MOD + RBD_ANY, re.I),
    re.compile(RBD_ANY + r'[^.;]{0,30}?(?:is|was|were)\s+(?:highly\s+)?'
               r'(?:suspected|noted|noticed|observed|recorded|confirmed|present)', re.I),
    re.compile(r'\bfindings?\s+(?:of|indicat\w+)\s+' + RBD_ANY, re.I),
    re.compile(r'^\s*' + RBD_ANY + r'\s*[:：(]', re.I),
]
BY_QUESTIONNAIRE = re.compile(r'(?:according\s+to|based\s+on|by)\s+[^.;]{0,40}?(?:<QSTN>|questionnaire|history)',
                              re.I)
_CLAUSE_SPLIT = re.compile(
    r'(?<=[.;])\s+|\s+(?=\d\.\s)|\b(?:but|however|whereas|although|though)\s+|,\s*but\s+', re.I)
STATUSES = ("definite", "probable", "pseudo_only", "indeterminate", "negative")


def classify_rbd(impression):
    """(status, first clause supporting it)."""
    t = RBDSQ_LONG.sub(' <QSTN> ', impression)
    t = QUESTIONNAIRE.sub(' <QSTN> ', t)
    t = RBDSS.sub(' <RBDSS> ', t)
    found = {s: [] for s in STATUSES}
    for cl in (p.strip() for p in _CLAUSE_SPLIT.split(t) if p and p.strip()):
        if not RBD_TOKEN.search(cl):
            continue
        has_pseudo = bool(PSEUDO.search(cl))
        cl2 = PSEUDO.sub(' <PSEUDO> ', cl)
        if any(p.search(cl2) for p in INDETERMINATE):        # "no REM sleep ... RBD cannot be determined"
            found["indeterminate"].append(cl)
        elif any(p.search(cl) for p in NEGATIVE):
            found["negative"].append(cl)
        elif has_pseudo and not RBD_TOKEN.search(cl2):
            found["pseudo_only"].append(cl)
        elif any(p.search(cl2) for p in POSSIBLE):
            found["probable"].append(cl)
        elif any(p.search(cl2) for p in POSITIVE):
            found["probable" if (BY_QUESTIONNAIRE.search(cl2) or "<QSTN>" in cl2) else "definite"].append(cl)
        else:
            found["probable"].append(cl)
    for s in STATUSES:
        if found[s]:
            return s, found[s][0][:400]
    return "none", ""


# RBDSQ score written in the narrative part of the impression (text before the first run of table cells)
RBDSQ = re.compile(r"RBD\s*-?\s*SQ\s*(?:was\s*)?[:：;]?\s*(-?\d+(?:\.\d+)?)", re.I)


def narrative(text):
    for m in re.finditer(r"\|\|", text):
        if text.count("||", m.start(), m.start() + 250) >= 3:
            return text[:m.start()]
    return text


def rbdsq(impression):
    m = RBDSQ.search(narrative(impression))
    return float(m.group(1)) if m else None
