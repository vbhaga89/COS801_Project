"""Validation-only pruning selection; ineligibility is not an execution failure."""
class PruningAccuracyGateError(RuntimeError):
    pass


def choose_pruning(candidates, allowed):
    if not candidates:
        raise ValueError('No pruning candidates supplied')
    eligible=[row for row in candidates if float(row['validation_accuracy_drop'])<=allowed]
    if not eligible:
        raise PruningAccuracyGateError('No pruning candidate satisfies the validation accuracy budget')
    return max(eligible,key=lambda row:(float(row['pruning_ratio']),row['validation_metrics']['macro_f1']))
