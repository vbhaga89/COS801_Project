"""Validation-tuned frozen ImageNet feature baseline; test evaluated once after selection."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', choices=['mobilenet_v3_small', 'efficientnet_b0'], required=True)
    args = parser.parse_args()
    import torch
    from rq71.data import build_dataloaders
    from rq71.models import build_linear_probe
    from rq71.engine import train_supervised, evaluate
    from run_experiment import set_seed
    config = json.loads(args.config.read_text())
    out = Path(config['output_dir']) / args.model / 'linear_probe'
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid = config.get('probe_learning_rates', [0.001, 0.01, 0.1])
    epochs = int(config.get('probe_epochs', 10))
    trials, best, best_score = [], None, -1
    for lr in grid:
        set_seed(int(config['seed']))
        loaders, classes = build_dataloaders(config)
        model = build_linear_probe(args.model, len(classes))
        model, history = train_supervised(model, loaders['train'], loaders['val'], device,
            epochs, lr, float(config['weight_decay']), config.get('max_train_batches'), config.get('max_eval_batches'))
        metrics = evaluate(model, loaders['val'], device, config.get('max_eval_batches'))
        trials.append(dict(learning_rate=lr, validation=metrics, history=history))
        if metrics['macro_f1'] > best_score:
            best_score = metrics['macro_f1']
            best = copy.deepcopy(model).cpu()
            selected_lr = lr
    # Rebuild only the loaders; no test feedback was used to pick the trial.
    loaders, classes = build_dataloaders(config)
    test = evaluate(best.to(device), loaders['test'], device, config.get('max_eval_batches'))
    torch.save(best.cpu().state_dict(), out / 'selected.pt')
    result = dict(seed=config['seed'], model=args.model, selected_learning_rate=selected_lr,
                  epochs_per_trial=epochs, trials=trials, test=test, classes=classes,
                  selection='validation macro F1; first trial wins ties', backbone_frozen=True)
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(test), flush=True)


if __name__ == '__main__':
    main()
