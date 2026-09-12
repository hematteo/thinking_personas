"""Two-hour Slurm controller with isolated seed workers and explicit cutoffs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

START=time.time()
CONFIG=os.environ.get('PERSONA_CONFIG','configs/followup_two_hour.json')
PYTHON=sys.executable
LOGS=Path('logs/followup');LOGS.mkdir(parents=True,exist_ok=True)
SPEC=json.loads(Path(CONFIG).read_text());ROOT=Path(SPEC['output_dir'])
TIMINGS=[]

def stop(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()


def stage(label,workers,cutoff):
    print(f'START {label} elapsed={(time.time()-START)/60:.2f} minutes',flush=True)
    handles=[];begun=time.time()
    try:
        for args,devices,suffix in workers:
            log=(LOGS/f'{label}-{suffix}.log').open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices)
            command=[PYTHON,'-m','persona_dynamics.followup','--config',CONFIG,*args]
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            handles.append((process,log,suffix))
        while True:
            states=[p.poll() for p,_,_ in handles]
            if any(code is not None and code!=0 for code in states):raise RuntimeError(f'{label} failed: {states}')
            if all(code==0 for code in states):break
            if time.time()>START+60*cutoff:raise TimeoutError(f'{label} reached {cutoff}-minute cutoff')
            time.sleep(2)
    finally:
        for p,log,_ in handles:stop(p);log.close()
    TIMINGS.append({'stage':label,'seconds':time.time()-begun,'elapsed_minutes':(time.time()-START)/60})
    (LOGS/'timings.json').write_text(json.dumps(TIMINGS,indent=2)+'\n')
    print(f'END {label}: {TIMINGS[-1]}',flush=True)


def main():
    try:
        stage('freeze',[(['--stage','init'],'','main')],5)
        stage('pilot-generation',[(['--stage','generate-pilot'],'0,1,2,3','pilot')],15)
        stage('validate',[(['--stage','validate'],'4,5,6,7','pilot')],25)
        stage('main-generation',[(['--stage','generate','--seed',str(seed)],devices,str(seed)) for seed,devices in zip(SPEC['seeds'],['0,1,2,3','4,5,6,7'])],85)
        stage('extraction',[(['--stage','extract','--seed',str(seed)],devices,str(seed)) for seed,devices in zip(SPEC['seeds'],['0,1,2,3','4,5,6,7'])],105)
        stage('analysis',[(['--stage','analyze'],'','main')],115)
        subprocess.run([PYTHON,'scripts/verify_followup.py',str(ROOT)],check=True,timeout=max(1,START+117*60-time.time()))
        status={'status':'complete','elapsed_seconds':time.time()-START,'start_unix':START,'deadline_unix':START+7200}
        print('COMPLETE '+json.dumps(status),flush=True)
    except BaseException as exc:
        status={'status':'failed','error':repr(exc),'elapsed_seconds':time.time()-START,'start_unix':START,'deadline_unix':START+7200}
        print(json.dumps(status),flush=True)
        ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'execution.json').write_text(json.dumps(status,indent=2)+'\n')
        raise
    (ROOT/'execution.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
