"""Shared catalog definitions for the ShoeLens application."""
import csv
import re
from pathlib import Path

FIELDS = ('articleType', 'baseColour', 'gender', 'season', 'usage')
ROOT = Path(__file__).resolve().parent


def load_catalog(path):
    """Load valid footwear records while reporting malformed or duplicate rows."""
    rows, rejected, seen = [], [], set()
    with open(path, newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        required = {'id', 'masterCategory', *FIELDS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing columns: ' + str(required - set(reader.fieldnames or [])))
        for line, row in enumerate(reader, 2):
            if None in row and reader.fieldnames[-1] == 'productDisplayName' and isinstance(row[None], list) and all(v is not None for k, v in row.items() if k is not None):
                row['productDisplayName'] += ',' + ','.join(row.pop(None))
            if None in row or any(v is None for v in row.values()):
                rejected.append({'line': line, 'reason': 'malformed CSV'})
                continue
            if row['masterCategory'].strip().lower() != 'footwear':
                continue
            product_id = row['id'].strip()
            if not re.fullmatch(r'[A-Za-z0-9_-]+', product_id) or product_id in seen:
                rejected.append({'line': line, 'reason': 'invalid/duplicate ID'})
                continue
            seen.add(product_id)
            rows.append({key: row.get(key, '').strip() for key in ('id', 'productDisplayName', 'masterCategory', *FIELDS)})
    return rows, rejected
