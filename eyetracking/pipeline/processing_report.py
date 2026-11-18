"""Processing-issue reporting for the stream pipeline.

Collects a per-participant outcome record during a pipeline run and writes two
files into the Results directory:
  - a plain-text summary (skips/errors grouped, pre-known issues surfaced), and
  - a per-participant CSV.

Each participant is cross-referenced against the "Bemerkungen" (remarks) and
"Dropout" columns of the lab-internal participant sheet (optional, not public), so issues that were
already documented during data collection sit next to what the pipeline
actually observed.

Sheet layout (sheet 'Study 1_Eye-tracker_EEG', header on row 0):
  col 0  = participant id           col 13 = Bemerkungen
  col 31 = Dropout
The main study rows number the subjects 1..100 in col 0. A leading pilot/redo
block ('TR 1'..'TR 6', plus duplicate 40/44/62/96 and '096 (neu)') precedes
them; 'TR n' rows are pilot test runs and are excluded, while a numeric id that
appears twice has its remarks merged (concatenated) so nothing is lost.
"""
import os
import re
import csv
from datetime import datetime


SHEET_NAME = 'Study 1_Eye-tracker_EEG'
COL_ID = 0
COL_BEMERKUNG = 13
COL_DROPOUT = 31

# Substrings (case-insensitive, German + English) that mark a remark as a
# likely data-quality problem, used only to decide which *completed*
# participants to surface in the text summary. Heuristic: the full remark is
# always kept in the CSV. Benign notes (calibration point counts, "Perfekt!")
# are intentionally not matched.
PROBLEM_KEYWORDS = (
    'abgebroch', 'abbruch', 'crash', 'gecrash', 'fehlt', 'fehlen', 'fehlend',
    'nicht funktioniert', 'funktioniert nicht', 'not work', 'kein', 'no data',
    'no eye', 'nicht gestartet', 'nicht gespeichert', 'not save', 'interrupt',
    'unterbroch', 'forgot', 'vergessen', 'verrutsch', 'wackelkontakt',
    'abgedriftet', 'zu hohe werte', 'zu hihe werte', 'sprünge', 'probleme',
)


def _is_concerning(bemerkung, dropout):
    """Heuristic: does this participant's sheet note flag a possible data issue?"""
    if dropout and str(dropout).strip().lower() not in ('', 'nein', 'no', 'nan'):
        return True
    text = (bemerkung or '').lower()
    return any(k in text for k in PROBLEM_KEYWORDS)


STAGE_LABELS = {
    'resume': 'skipped (resume: Phase 5 already done)',
    'phase1_extract': 'skipped at Phase 1 - no raw data extracted (e.g. no fixation events)',
    'phase2_chunk': 'skipped at Phase 2 - no movie chunks (e.g. screenshot session / no movie timestamps)',
    'phase3_frames': 'skipped at Phase 3 - no frame matches',
    'phase4_masks': 'skipped at Phase 4 - no segmentation-mask matches',
    'error': 'ERROR - unhandled exception (see run log)',
    'completed': 'completed',
    'completed_cache': 'completed (from cached pkl)',
}


def participant_number(participant_id):
    """Map a pipeline participant id ('039_02') to an int subject number (39)."""
    m = re.match(r'0*(\d+)', str(participant_id).strip())
    return int(m.group(1)) if m else None


def session_of(participant_id):
    """Return the session suffix ('01'/'02') from a participant id, or ''."""
    m = re.search(r'_(\d+)\s*$', str(participant_id).strip())
    return m.group(1) if m else ''


def _parse_sheet_id(raw):
    """Resolve a sheet col-0 label to an int subject number, or None.

    'TR 1'..'TR 6' are pilot test runs (excluded). '096 (neu)' -> 96.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower().startswith('tr'):
        return None
    m = re.match(r'^0*(\d+)', s)
    return int(m.group(1)) if m else None


def load_participant_remarks(xlsx_path, logger=None):
    """Return {subject_number: {'bemerkung', 'dropout', 'n_rows'}} from the sheet.

    Multiple sheet rows for the same id are merged: remarks are concatenated
    with ' | ' so nothing is lost. Returns {} (with a warning) if the sheet
    can't be read.
    """
    remarks = {}
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME, header=None)
    except Exception as e:
        if logger:
            logger.warning('Could not read participant sheet %s: %s. '
                           'Report will omit Bemerkungen.', xlsx_path, e)
        return remarks

    import pandas as pd  # local import so the module loads without pandas present
    for i in range(1, len(df)):          # row 0 is the header
        sid = _parse_sheet_id(df.iat[i, COL_ID])
        if sid is None:
            continue
        rec = remarks.setdefault(sid, {'bemerkung': '', 'dropout': '', 'n_rows': 0})
        rec['n_rows'] += 1
        for field, col in (('bemerkung', COL_BEMERKUNG), ('dropout', COL_DROPOUT)):
            val = df.iat[i, col]
            val = '' if pd.isna(val) else str(val).strip().replace('\n', ' ')
            if not val or val.lower() == 'nan':
                continue
            if not rec[field]:
                rec[field] = val
            elif val not in rec[field]:
                rec[field] = f'{rec[field]} | {val}'
    return remarks


def _enrich(records, remarks):
    """Attach subject number, session and sheet remarks to each record."""
    rows = []
    for r in records:
        sid = participant_number(r['participant_id'])
        rem = remarks.get(sid, {})
        rows.append({
            'participant_id': r['participant_id'],
            'subject': sid,
            'session': r.get('session', session_of(r['participant_id'])),
            'outcome': r['outcome'],
            'stage': r['stage'],
            'reason': r.get('reason', ''),
            'bemerkung': rem.get('bemerkung', ''),
            'dropout': rem.get('dropout', ''),
            'sheet_rows_merged': rem.get('n_rows', 0),
        })
    return rows


def write_processing_report(records, remarks, out_dir, timestamp=None, logger=None):
    """Write the CSV + txt processing report. Returns (csv_path, txt_path)."""
    os.makedirs(out_dir, exist_ok=True)
    ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'pipeline_processing_report_{ts}.csv')
    txt_path = os.path.join(out_dir, f'pipeline_processing_report_{ts}.txt')

    rows = _enrich(records, remarks)

    fields = ['participant_id', 'subject', 'session', 'outcome', 'stage',
              'reason', 'bemerkung', 'dropout', 'sheet_rows_merged']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    completed = [r for r in rows if r['outcome'] == 'completed']
    skipped = [r for r in rows if r['outcome'] == 'skipped']
    errored = [r for r in rows if r['outcome'] == 'error']

    def _fmt(r):
        line = f"  {r['participant_id']:12} {STAGE_LABELS.get(r['stage'], r['stage'])}"
        if r['reason']:
            line += f"\n      reason: {r['reason']}"
        if r['bemerkung']:
            line += f"\n      sheet Bemerkung: {r['bemerkung']}"
        if r['dropout']:
            line += f"\n      sheet Dropout: {r['dropout']}"
        return line

    lines = []
    lines.append('=' * 70)
    lines.append('PIPELINE PROCESSING REPORT')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * 70)
    lines.append('')
    lines.append(f'Participants attempted : {len(rows)}')
    lines.append(f'  completed            : {len(completed)}')
    lines.append(f'  skipped              : {len(skipped)}')
    lines.append(f'  errored              : {len(errored)}')
    lines.append('')

    lines.append('-' * 70)
    lines.append(f'ERRORED ({len(errored)}) - unhandled exceptions, see run log')
    lines.append('-' * 70)
    for r in errored:
        lines.append(_fmt(r))
    if not errored:
        lines.append('  (none)')
    lines.append('')

    lines.append('-' * 70)
    lines.append(f'SKIPPED ({len(skipped)})')
    lines.append('-' * 70)
    for r in sorted(skipped, key=lambda x: x['stage']):
        lines.append(_fmt(r))
    if not skipped:
        lines.append('  (none)')
    lines.append('')

    # Completed but pre-flagged in the sheet: process succeeded, yet the
    # remarks document a possible data-quality caveat worth a manual look.
    # Filtered by keyword so benign calibration notes don't flood the summary.
    flagged_ok = [r for r in completed if _is_concerning(r['bemerkung'], r['dropout'])]
    lines.append('-' * 70)
    lines.append(f'COMPLETED WITH POSSIBLE DATA ISSUES ({len(flagged_ok)}) - heuristic, review manually')
    lines.append('-' * 70)
    for r in sorted(flagged_ok, key=lambda x: (x['subject'] or 0)):
        lines.append(_fmt(r))
    if not flagged_ok:
        lines.append('  (none)')
    lines.append('')

    # Ids whose sheet remarks were merged from >1 row (pilot/redo duplicates):
    # attribution is uncertain, so list them for verification.
    merged = sorted({r['subject'] for r in rows if r['sheet_rows_merged'] and r['sheet_rows_merged'] > 1})
    if merged:
        lines.append('-' * 70)
        lines.append('NOTE: sheet remarks merged from multiple rows for subjects: '
                     + ', '.join(str(m) for m in merged))
        lines.append('  (a pilot/redo row plus the main row share an id; verify attribution)')
        lines.append('')

    lines.append('=' * 70)
    lines.append('END OF REPORT')
    lines.append('=' * 70)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    if logger:
        logger.info('Processing report written: %s', txt_path)
        logger.info('Processing report (CSV):   %s', csv_path)
    return csv_path, txt_path
