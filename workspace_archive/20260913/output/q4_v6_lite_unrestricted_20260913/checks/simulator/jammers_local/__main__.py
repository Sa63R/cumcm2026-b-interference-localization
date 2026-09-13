"""CLI: serve, generate, batch and deterministic replay."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from .core import Scenario, Session, LocalClient
from .baseline import run as baseline


def dump(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load_policy(spec):
    if spec=='baseline': return baseline
    filename,separator,function=spec.rpartition(':')
    if not separator: raise ValueError('strategy must be baseline or /path/to/policy.py:function')
    path=Path(filename).resolve()
    module_spec=importlib.util.spec_from_file_location('user_jammers_policy',path)
    if module_spec is None or module_spec.loader is None: raise ValueError('cannot load strategy')
    module=importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name]=module
    module_spec.loader.exec_module(module)
    result=getattr(module,function)
    if not callable(result): raise ValueError('strategy entry is not callable')
    return result


def replay(data):
    if data.get('schema_version')!='local-jammers-run-v1': raise ValueError('expected exported run JSON')
    session=Session(Scenario.from_dict(data['state']['scenario']),robot_id=data['state']['robot_id'],**data.get('limits',{}))
    mismatches=[]
    for record in data['history']:
        status,actual=session.dispatch(record['path'],record['request'])
        expected={k:v for k,v in record['response'].items() if k not in {'real_timestamp_ms','remaining_real_duration_s'}}
        comparable={k:v for k,v in actual.items() if k not in {'real_timestamp_ms','remaining_real_duration_s'}}
        if status!=200 or expected!=comparable:
            mismatches.append(dict(index=record['index'],expected=expected,actual=comparable,status=status))
    return dict(matched=not mismatches,compared_actions=len(data['history']),mismatches=mismatches,
                virtual_time_s=session.engine.time_us/1_000_000)


def main(argv=None):
    p=argparse.ArgumentParser(description='干扰源本地模拟器：基于exe静态还原，无需Windows或账号')
    sub=p.add_subparsers(dest='command',required=True)
    serve=sub.add_parser('serve',help='启动本地HTTP接口和网页测试台')
    serve.add_argument('--port',type=int,default=2027)
    serve.add_argument('--robot-id',default='local-test')
    serve.add_argument('--problem',type=int,choices=[3,4],default=4)
    serve.add_argument('--seed',default='demo-20260912')
    serve.add_argument('--seed-hex')
    serve.add_argument('--scenario',type=Path)
    serve.add_argument('--results',type=Path,default=Path('results/server'))
    gen=sub.add_parser('generate',help='导出固定种子的场景真值JSON')
    gen.add_argument('--problem',type=int,choices=[3,4],default=4)
    gen.add_argument('--seed',default='demo-20260912')
    gen.add_argument('--seed-hex')
    gen.add_argument('--output',type=Path,required=True)
    batch=sub.add_parser('batch',help='通过同一核心执行离线策略批量评测')
    batch.add_argument('--problem',type=int,choices=[3,4],default=4)
    batch.add_argument('--count',type=int,default=10)
    batch.add_argument('--start-seed',type=int,default=1)
    batch.add_argument('--strategy',default='baseline')
    batch.add_argument('--output',type=Path,default=Path('results/batch'))
    rep=sub.add_parser('replay',help='重放导出的已接受动作，核对业务响应及微秒时钟')
    rep.add_argument('run',type=Path)
    args=p.parse_args(argv)
    try:
        if args.command=='serve':
            from .server import Application,Server
            scenario=Scenario.from_dict(json.loads(args.scenario.read_text())) if args.scenario else Scenario.generate(args.problem,args.seed,seed_hex=args.seed_hex)
            app=Application(scenario,args.robot_id,results_dir=args.results)
            with Server(app,args.port) as server:
                print(f'本地测试台：http://127.0.0.1:{server.server_port}\nrobot_id：{args.robot_id}\nCtrl+C 停止；结果目录：{args.results.resolve()}',flush=True)
                try: server.serve_forever()
                except KeyboardInterrupt: pass
                finally:
                    app.cancel.set()
                    if app.demo_thread: app.demo_thread.join(timeout=3)
                    app.save()
        elif args.command=='generate':
            scene=Scenario.generate(args.problem,args.seed,seed_hex=args.seed_hex)
            dump(args.output,scene.as_dict())
            print(f'已保存：{args.output.resolve()}（案例 {scene.case_id}）')
        elif args.command=='replay':
            result=replay(json.loads(args.run.read_text()))
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result['matched'] else 1
        else:
            if not 1<=args.count<=10000: raise ValueError('count must be 1..10000')
            # Refuse accidental overwrite of an earlier comparison.
            if args.output.exists() and any(args.output.iterdir()): raise ValueError('输出目录非空，请指定新的 --output 目录')
            args.output.mkdir(parents=True,exist_ok=True)
            policy=load_policy(args.strategy)
            rows=[]
            for i in range(args.count):
                seed=str(args.start_seed+i)
                session=Session(Scenario.generate(args.problem,seed))
                client=LocalClient(session.dispatch)
                error=None
                try: policy(client,args.problem)
                except Exception as exc: error=f'{type(exc).__name__}: {exc}'
                if session.engine.lifecycle!='ended': session.engine.finish('harness_finished')
                result=session.export()
                result['policy_error']=error
                dump(args.output/f'case-{i+1:04d}.json',result)
                row={k:v for k,v in result['state'].items() if k not in {'scenario'}}
                row.update(seed=seed,policy_error=error)
                rows.append(row)
                print(f'{i+1}/{args.count} seed={seed} cleared={row["cleared_count"]}/{row["source_total"]} virtual={row["virtual_time_s"]:.3f}s'+(f' error={error}' if error else ''),flush=True)
            summary=dict(kind='local_static_reconstruction',problem=args.problem,strategy=args.strategy,
                         case_count=len(rows),all_cleared_cases=sum(r['all_cleared'] for r in rows),
                         policy_errors=sum(r['policy_error'] is not None for r in rows),
                         mean_virtual_time_s=sum(r['virtual_time_s'] for r in rows)/len(rows),
                         cases=rows)
            dump(args.output/'summary.json',summary)
            print(f'汇总：{(args.output/"summary.json").resolve()}',flush=True)
            return 1 if summary['policy_errors'] else 0
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        p.exit(1,f'错误：{exc}\n')
    return 0


if __name__=='__main__': raise SystemExit(main())
