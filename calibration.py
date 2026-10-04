"""Choose an empirical acceptance gate using calibration predictions only."""


def select_threshold(rows, target=.85, min_accepted=10):
    if not rows:
        raise ValueError('Calibration rows required')
    best = None
    for threshold in sorted({row['score'] for row in rows}):
        for margin in sorted({0.000001, *[row['margin'] for row in rows if row['margin'] > 0]}):
            accepted = [row for row in rows if row['is_footwear'] and row['score'] >= threshold and row['margin'] >= margin]
            if len(accepted) < min_accepted:
                continue
            precision = sum(row['correct'] for row in accepted) / len(accepted)
            if precision >= target and (best is None or len(accepted) > best['accepted']):
                best = {'threshold': threshold, 'margin': margin, 'precision': precision, 'accepted': len(accepted)}
    if best is None:
        return {'threshold': 1.000001, 'margin': 1.000001, 'precision': None, 'accepted': 0}
    return best
