import pandas as pd
import re

df = pd.read_excel(r'C:\Users\SujitR\Desktop\rfai\5G_BL.xlsx')

print(f"Loaded {len(df)} parameters")
print(f"Columns: {list(df.columns[:5])}")

# ── Clean Abbreviated Name ─────────────────────────────────
# Remove Nokia MO notation like "a3MeasSsbRsrp.[all]."
# Keep only the actual parameter name part after the last dot
def clean_abbrev(val):
    if pd.isna(val):
        return val
    val = str(val)
    # Remove [all] notation
    val = val.replace('.[all]', '')
    # Keep only last part after final dot
    parts = val.split('.')
    if len(parts) > 1:
        return parts[-1]
    return val

df['Abbreviated Name'] = df['Abbreviated Name'].apply(clean_abbrev)

# ── Clean MO Class ─────────────────────────────────────────
# Replace Nokia-specific MO class names with generic 3GPP names
mo_map = {
    'NRCELL':  'NR Cell',
    'NRHOIF':  'NR HO Interface',
    'NRBTS':   'NR Base Station',
    'NRDU':    'NR DU',
    'NRCU':    'NR CU',
}
df['MO Class'] = df['MO Class'].apply(
    lambda x: mo_map.get(str(x).strip(), str(x)) if pd.notna(x) else x
)

# ── Shorten Description ────────────────────────────────────
# Keep only first sentence — remove Nokia-specific references
def shorten_description(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Keep first sentence only
    sentences = re.split(r'(?<=[.!?])\s+', val)
    short = sentences[0] if sentences else val
    # Remove Nokia feature references like [38.321], 5GCxxxxxx
    short = re.sub(r'\[[\d.]+\]', '', short)
    short = re.sub(r'5GC\d+[:\s]*\w*', '', short)
    short = short.strip()
    # Truncate if still long
    if len(short) > 150:
        short = short[:147] + '...'
    return short

df['Description'] = df['Description'].apply(shorten_description)

# ── Remove Nokia-specific columns ─────────────────────────
# These columns contain proprietary workflow info
cols_to_blank = [
    'RAC DB table name', 'RAC DB column name',
    'SON Equation', 'LastModifiedUser',
    'SON Audit Rule', 'SON Enforce this Parameter',
    'SON Enforcement Rule', 'SON MW Enforcement Only',
    'SON Regional Level Enforcement',
    'SON Market Level Enforcement',
    'SON Allowable Enforcement Values',
    'Comments SON', 'Audit Type'
]
for col in cols_to_blank:
    if col in df.columns:
        df[col] = ''

# ── Remove Nokia feature references from Comments ──────────
def clean_comments(val):
    if pd.isna(val):
        return val
    val = str(val)
    # Remove 5GCxxxxxx feature references
    val = re.sub(r'5GC\d+[:\s]*[^\n]*', '', val)
    val = val.strip()
    return val if val else ''

df['Comments'] = df['Comments'].apply(clean_comments)

# ── Remove Nokia feature references from Related Features ──
if 'Related Features' in df.columns:
    df['Related Features'] = df['Related Features'].apply(
        lambda x: re.sub(r'5GC\d+[^\n]*', '', str(x)).strip() 
        if pd.notna(x) else x
    )

# ── Save anonymized version ────────────────────────────────
output_path = r'C:\Users\SujitR\Desktop\rfai\5G_BL_public.xlsx'
df.to_excel(output_path, index=False)
print(f"\nSaved anonymized baseline to: 5G_BL_public.xlsx")

# Show sample of changes
print("\nSample after anonymization:")
sample_cols = ['Full Name', 'Abbreviated Name', 'MO Class',
               'Description', 'Operator recommended value', 'Range and step']
print(df[sample_cols].head(3).to_string())