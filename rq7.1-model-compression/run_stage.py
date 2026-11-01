"""Independent notebook stages over an explicit resolved seed configuration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def load_stage_config(path):
    config=json.loads(Path(path).read_text())
    config['_config_path']=str(Path(path).resolve())
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['baselines','compression','evaluation'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='Explicit synthetic integration test only')
    args=parser.parse_args()
    from run_three_seeds import validate_config, verify_data
    cfg=load_stage_config(args.config)
    if args.smoke:
        if cfg.get('smoke_mode') is not True or cfg.get('batch_size')!=64:
            raise ValueError('Smoke requires explicit smoke_mode=true and batch_size=64')
    else:
        if cfg.get('smoke_mode'):raise ValueError('Smoke configuration cannot run as official')
        validate_config(cfg)
    root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=True)
    statefile=root/'notebook_stage_state.json'
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    state=json.loads(statefile.read_text()) if statefile.exists() else dict(config_sha256=signature,stages={})
    if state['config_sha256']!=signature:raise ValueError('Configuration changed between notebook stages')
    if args.stage in state['stages']:raise RuntimeError('Stage already attempted: inspect evidence and use a fresh run; no silent overwrite/resume')
    previous={'compression':'baselines','evaluation':'compression'}.get(args.stage)
    if previous and state['stages'].get(previous) not in ('complete','partial'):raise RuntimeError(f'Complete {previous} first')
    def save():statefile.write_text(json.dumps(state,indent=2),encoding='utf-8')
    state['stages'][args.stage]='running';save()
    try:
        import run_experiment as runner
        models=('mobilenet_v3_small','efficientnet_b0')
        if args.stage=='baselines':
            if not args.smoke:verify_data(Path(cfg['data_dir']),Path(cfg['manifest_dir']))
            subprocess.run([sys.executable,str(Path(__file__).parent/'check_environment.py'),
                            *([] if args.smoke else ['--require-cuda']),
                            '--output',str(root/'environment.json')],check=True)
            runner.command_eda(cfg)
            for model in models:
                subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'linear_probe.py'),'--config',str(args.config.resolve()),'--model',model],check=True)
                runner.set_seed(int(cfg['seed']));runner.command_train(cfg,model)
        elif args.stage=='compression':
            failures=[]
            for model in models:
                checkpoint=str(root/model/'baseline.pt')
                runner.set_seed(int(cfg['seed']))
                try:runner.command_quantize(cfg,checkpoint)
                except runner.QuantizationAccuracyGateError as exc:failures.append(dict(method='int8',model=model,status='ineligible_accuracy_gate',error=str(exc),test_evaluated=False))
                for ratio in cfg['pruning_ratios']:
                    candidate=dict(cfg,pruning_ratio=ratio)
                    runner.set_seed(int(cfg['seed']));runner.command_prune(candidate,checkpoint)
                runner.set_seed(int(cfg['seed']))
                try:runner.command_prune_select(cfg,checkpoint)
                except runner.PruningAccuracyGateError as exc:failures.append(dict(method='pruning',model=model,status='ineligible_accuracy_gate',error=str(exc),test_evaluated=False))
                runner.set_seed(int(cfg['seed']));runner.command_distill(cfg,checkpoint)
            state['accuracy_gate_failures']=failures
        else:
            runner.command_report(cfg)
            subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'analyse_selected_models.py'),'--config',str(args.config.resolve())],check=True)
        state['stages'][args.stage]='partial' if args.stage=='compression' and state.get('accuracy_gate_failures') else 'complete'
        save()
        if args.stage=='evaluation':
            manifest=dict(seed=cfg['seed'],status='partial' if state.get('accuracy_gate_failures') else 'complete',
                          smoke_mode=bool(args.smoke),
                          stages=[dict(stage=k,status=v) for k,v in state['stages'].items()],
                          incomplete_stages=state.get('accuracy_gate_failures',[]),entrypoint='independent notebook stages')
            (root/'full_run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    except Exception as exc:
        state['stages'][args.stage]='failed';state['error']=f'{type(exc).__name__}: {exc}';save();raise
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
