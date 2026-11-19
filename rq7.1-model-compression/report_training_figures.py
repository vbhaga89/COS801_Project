"""Render appendix-style comparisons from saved histories without training."""
import argparse
import json
from pathlib import Path


def render(root):
    from matplotlib import pyplot as plt
    destination = root / 'figures'
    destination.mkdir(exist_ok=True)
    def read(path):
        return json.loads(path.read_text()) if path.is_file() else None
    for kind in ('baselines', 'pruning', 'student_ce', 'kd_objective'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for model, ax in zip(('mobilenet_v3_small', 'efficientnet_b0'), axes):
            if kind == 'baselines':
                history = read(root/model/'baseline_history.json')
                if history:
                    x = range(1, len(history)+1)
                    for field, label in [('train_loss','Train CE'),('loss','Validation CE')]:
                        if field in history[0]: ax.plot(x,[r[field] for r in history],label=label)
                    for i in range(1,len(history)):
                        if history[i].get('stage') != history[i-1].get('stage'):
                            ax.axvline(i+.5,color='gray',linestyle=':')
            elif kind == 'pruning':
                for ratio in (.1,.2,.3):
                    history=read(root/model/f'pruning_history_r{ratio:.2f}.json')
                    if history:
                        for field, style in [('train_loss','-'),('loss','--')]:
                            if field in history[0]: ax.plot(range(1,len(history)+1),[r[field] for r in history],style,label=f'{ratio:.0%} {field}')
            else:
                folder=root/'mobilenet_v3_small_050'/f'teacher_{model}'
                series=([('distillation_history.json','train_loss','KD objective')] if kind=='kd_objective' else [('student_control_history.json','train_loss','CE control train CE'),('student_control_history.json','loss','CE control val CE'),('distillation_history.json','loss','KD student val CE')])
                for filename,field,label in series:
                    history=read(folder/filename)
                    if history and field in history[0]:ax.plot(range(1,len(history)+1),[r[field] for r in history],label=label)
            if ax.lines: ax.legend(fontsize=8)
            else: ax.text(.5,.5,'History unavailable',ha='center',transform=ax.transAxes)
            ax.set(title=model,xlabel='Sequential epoch',ylabel='Combined KD objective' if kind=='kd_objective' else 'Cross-entropy loss')
            ax.grid(alpha=.2)
        fig.suptitle(f'{root.name}: {kind}; saved histories only')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(destination/f'training_{kind}.{ext}',dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed_directory',type=Path)
    render(parser.parse_args().seed_directory)
