"""Independently verify raw experiment counts, arithmetic and reported metrics."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');a=p.parse_args()
    root=Path(a.results);project=root.parent
    tr=pd.read_csv(root/'training.csv');te=pd.read_csv(root/'test.csv')
    va=pd.read_csv(root/'validation.csv');ro=pd.read_csv(root/'robustness.csv')
    s=pd.read_csv(root/'summary.csv').set_index('config')
    rr=pd.read_csv(root/'robustness_summary.csv').set_index(['phase','noise'])
    checks=[]
    def check(condition,label):
        if not condition:raise AssertionError(label)
        checks.append(label)
    check(len(tr)==225000,'225000 main training episodes')
    check(len(te)==4500 and len(va)==4500,'4500 final test and 4500 validation episodes')
    check(len(ro)==2500,'2500 held-out robustness episodes')
    check(tr.groupby(['config','train_seed']).size().eq(5000).all(),'45 complete full-budget training runs')
    for label,data in [('training',tr),('test',te),('validation',va),('robustness',ro)]:
        check(np.array_equal(data.reward.to_numpy(),(-data.steps-99*data.cliff_falls).to_numpy()),
              f'{label}: every return equals -steps - 99*falls')
        check(((data.steps>=1)&(data.steps<=500)).all(),f'{label}: episode lengths in range')
        check(((data.success+data.truncated)==1).all(),f'{label}: episodes end at goal or cap')
        check((data.loc[data.truncated==1,'steps']==500).all(),f'{label}: truncations occur at 500')
    for name,d in te.groupby('config'):
        seed=d.groupby('train_seed').reward.mean()
        check(np.isclose(seed.mean(),s.loc[name,'test_mean']),f'{name}: test mean recomputed')
        check(np.isclose(seed.std(),s.loc[name,'test_seed_sd']),f'{name}: seed SD recomputed')
        qfiles=list((root/'runs'/name).glob('seed_*/q_table.npy'))
        check(len(qfiles)==5 and all(np.load(f).shape==(48,4) and np.isfinite(np.load(f)).all() for f in qfiles),
              f'{name}: five finite 48 by 4 Q-tables')
        if not name.startswith('noise_'):
            check((d.reward==-13).all() and (d.steps==13).all() and (d.success==1).all(),f'{name}: optimal deterministic tests')
    for key,d in ro.groupby(['phase','eval_noise']):
        means=d.groupby('train_seed')[['reward','steps','cliff_falls','success','truncated']].mean()
        for col in means:
            check(np.isclose(means[col].mean(),rr.loc[key,col+'_mean']),f'{key}: {col} mean')
            check(np.isclose(means[col].std(),rr.loc[key,col+'_seed_sd']),f'{key}: {col} SD')
    deterministic=[n for n in s.index if not n.startswith('noise_')]
    cp=pd.read_csv(root/'checkpoints.csv')
    ranking=va[va.config.isin(deterministic)].groupby('config').reward.mean().rename('validation_reward').to_frame()
    ranking['early_checkpoint_reward']=cp[(cp.config.isin(deterministic))&(cp.episode<=1000)].groupby('config').reward.mean()
    ranking['training_reward']=tr[tr.config.isin(deterministic)].groupby('config').reward.mean()
    ranking=ranking.reset_index().sort_values(['validation_reward','early_checkpoint_reward','training_reward','config'],ascending=[False,False,False,True])
    chosen=json.loads((root/'best_config.json').read_text())
    check(chosen['name']==ranking.iloc[0]['config']=='gamma_0.9','selection independently reproduced without test data')
    for n in deterministic:
        for seed in range(5):
            path=Path('runs')/n/f'seed_{seed}'/'training.csv'
            main_run=pd.read_csv(root/path).iloc[:200].reset_index(drop=True)
            replay=pd.read_csv(root/'diagnostics'/path)
            pd.testing.assert_frame_equal(main_run,replay)
    check(True,'all 35 fine-resolution replay prefixes exactly equal original training')
    result=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration,size',
            '-show_entries','stream=codec_name,width,height','-of','json',
            str(project/'videos/trained_agent.mp4')],check=True,capture_output=True,text=True)
    video=json.loads(result.stdout)
    check(float(video['format']['duration'])==10.0 and int(video['format']['size'])>1000,
          'video is a nonempty 10 second MP4')
    check(video['streams'][0]['codec_name']=='h264','video uses H264 codec')
    # Full decode catches a corrupt stream even if its container metadata is valid.
    subprocess.run(['ffmpeg','-v','error','-i',str(project/'videos/trained_agent.mp4'),'-f','null','-'],check=True)
    check(True,'all video frames decoded successfully')
    out={'status':'PASS','checks_passed':len(checks),'checks':checks,'video':video}
    (project/'verification.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'status':'PASS','checks_passed':len(checks),'main_training_runs':45,
                      'main_training_episodes':225000,'test_episodes':4500,
                      'robustness_episodes':2500},indent=2))


if __name__=='__main__':main()
