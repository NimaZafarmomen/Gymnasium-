"""Create tables, report figures and an MP4 from the saved experiment outputs."""
from pathlib import Path
import argparse
import json
import os
import subprocess
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches, font_manager
from PIL import Image, ImageDraw, ImageFont
from q_learning import Config, make_env

COLORS = ["#24618A", "#BD5C2B", "#29816B"]


def set_style():
    for path in Path('/usr/local/share/fonts/truetype/msttcorefonts').glob('*.TTF'):
        font_manager.fontManager.addfont(path)
    preferred = 'Times New Roman'
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": preferred if preferred in available else 'DejaVu Serif',
                         "font.size": 11, "axes.labelsize": 11,
                         "axes.titlesize": 11.5, "legend.fontsize": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "figure.dpi": 150})


def smooth_curve(data, name, window=100):
    d = data[data.config == name].pivot(index='episode', columns='train_seed', values='reward')
    d = d.rolling(window, min_periods=window).mean().dropna()
    return d.index.to_numpy(), d.mean(axis=1).to_numpy(), d.std(axis=1).to_numpy()


def band(ax, x, mean, sd, color, label):
    ax.plot(x, mean, color=color, label=label, lw=1.6)
    ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.13, linewidth=0)
    ax.grid(axis='y', alpha=.18)


def learning_figure(training, early, names, labels, output, title):
    fig, axs = plt.subplots(1, 2, figsize=(6.65, 3.1), layout='constrained')
    for name, label, color in zip(names, labels, COLORS):
        x, mean, sd = smooth_curve(training, name)
        band(axs[0], x, mean, sd, color, label)
        d = early[early.config == name].pivot(index='episode', columns='train_seed', values='reward')
        band(axs[1], d.index.to_numpy(), d.mean(axis=1).to_numpy(), d.std(axis=1).to_numpy(), color, label)
    axs[0].set(xlabel='Training episode', ylabel='100-episode mean return',
               title='Training with exploration', xlim=(0, 5000))
    axs[1].set(xlabel='Training episode', ylabel='Greedy validation return',
               title='Early policy learning', xlim=(0, 200))
    axs[1].axhline(-13, color='black', lw=.75, ls='--', label='Optimum −13')
    axs[0].legend(loc='lower right', frameon=False)
    axs[1].legend(loc='lower right', frameon=False)
    fig.savefig(output, bbox_inches='tight'); plt.close(fig)


def grid_figure(output):
    fig, ax = plt.subplots(figsize=(6.65, 2.4), layout='constrained')
    for r in range(4):
        for c in range(12):
            color = '#F2F2F2'
            if r == 3 and 1 <= c <= 10: color = '#ECD4C5'
            if (r,c) == (3,0): color = '#C8DFEB'
            if (r,c) == (3,11): color = '#C7E0D6'
            ax.add_patch(patches.Rectangle((c,r),1,1,facecolor=color,edgecolor='#AAAAAA',lw=.6))
            label = str(r*12+c)
            if (r,c) == (3,0): label='S\n36'
            if (r,c) == (3,11): label='G\n47'
            if r == 3 and 1 <= c <= 10: label='Cliff'
            ax.text(c+.5,r+.5,label,ha='center',va='center',fontsize=9 if label=='Cliff' else 10)
    ax.set(xlim=(0,12),ylim=(4,0),aspect='equal',xticks=[],yticks=[])
    for s in ax.spines.values(): s.set_visible(False)
    fig.savefig(output,bbox_inches='tight');plt.close(fig)


def route_figure(output):
    fig, ax = plt.subplots(figsize=(6.65, 2.1), layout='constrained')
    for r in range(4):
        for c in range(12):
            color='#F4F4F4'
            if r==3 and 1<=c<=10: color='#ECD4C5'
            ax.add_patch(patches.Rectangle((c,r),1,1,facecolor=color,edgecolor='#BBBBBB',lw=.6))
    positions=[(0.5,3.5)]+[(c+.5,2.5) for c in range(12)]+[(11.5,3.5)]
    for (x,y),(xx,yy) in zip(positions[:-1],positions[1:]):
        ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=COLORS[0],lw=1.8))
    ax.text(.5,3.8,'Start',ha='center',va='center',fontsize=9)
    ax.text(11.5,3.8,'Goal',ha='center',va='center',fontsize=9)
    ax.text(6,3.5,'Cliff',ha='center',va='center',fontsize=11)
    ax.set(xlim=(0,12),ylim=(4,0),aspect='equal',xticks=[],yticks=[])
    for s in ax.spines.values(): s.set_visible(False)
    fig.savefig(output,bbox_inches='tight');plt.close(fig)


def metrics(root, training, checkpoints, early, test, robustness):
    records=[]
    for name, d in training.groupby('config',sort=False):
        final = test[test.config == name]
        seed_means = final.groupby('train_seed')[['reward','steps','cliff_falls','success','truncated']].mean()
        first = d[d.episode<=500].groupby('train_seed').reward.mean()
        last = d[d.episode>4500].groupby('train_seed').reward.mean()
        sd_last=d[d.episode>4500].groupby('train_seed').reward.std()
        rec=dict(config=name,first500_mean=first.mean(),first500_sd=first.std(),
                 last500_mean=last.mean(),last500_sd=last.std(),
                 last500_episode_sd=sd_last.mean(),overall_training=d.reward.mean(),
                 test_mean=seed_means.reward.mean(),test_seed_sd=seed_means.reward.std(),
                 test_episode_sd=final.reward.std(),success=seed_means.success.mean(),
                 steps=seed_means.steps.mean(),cliff_falls=seed_means.cliff_falls.mean(),
                 truncation=seed_means.truncated.mean())
        if not name.startswith('noise_'):
            times=[]
            for seed in sorted(d.train_seed.unique()):
                e=early[(early.config==name)&(early.train_seed==seed)]
                c=checkpoints[(checkpoints.config==name)&(checkpoints.train_seed==seed)&(checkpoints.episode>200)]
                merged=pd.concat([e,c]).sort_values('episode')
                good=(merged.reward.to_numpy()==-13)
                persistent=np.logical_and.accumulate(good[::-1])[::-1]
                times.append(int(merged.episode.to_numpy()[np.flatnonzero(persistent)[0]]))
            rec.update(optimal_checkpoint_median=float(np.median(times)),
                       optimal_checkpoint_min=min(times),optimal_checkpoint_max=max(times),
                       optimal_checkpoints=times)
        records.append(rec)
    summary=pd.DataFrame(records)
    summary.to_csv(root/'summary.csv',index=False)
    robust_rows=[]
    for (phase,p),d in robustness.groupby(['phase','eval_noise']):
        s=d.groupby('train_seed')[['reward','steps','cliff_falls','success','truncated']].mean()
        row={'phase':phase,'noise':p,'n_episodes':len(d)}
        for col in s:
            row[col+'_mean']=float(s[col].mean());row[col+'_seed_sd']=float(s[col].std())
        row['reward_episode_sd']=float(d.reward.std())
        robust_rows.append(row)
    pd.DataFrame(robust_rows).to_csv(root/'robustness_summary.csv',index=False)
    (root/'summary.json').write_text(json.dumps(records,indent=2))
    return summary,pd.DataFrame(robust_rows)


def record_video(root, out):
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    best=Config(**json.loads((root/'best_config.json').read_text()))
    q=np.load(root/'runs'/best.name/'seed_0/q_table.npy')
    env=make_env(best,render_mode='rgb_array')
    state,_=env.reset(seed=300000)
    total=0;step=0
    frame=env.render()
    height,width=frame.shape[:2]
    h=height+88
    if h%2:h+=1
    out.parent.mkdir(exist_ok=True,parents=True)
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{width}x{h}','-r','8','-i','-','-an','-c:v','libx264',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    pipe=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    fontpath='/usr/local/share/fonts/truetype/msttcorefonts/Times.TTF'
    font=ImageFont.truetype(fontpath,24) if Path(fontpath).exists() else ImageFont.load_default()
    def add(frame,repeat):
        img=Image.new('RGB',(width,h),'white');img.paste(Image.fromarray(frame),(0,88))
        draw=ImageDraw.Draw(img)
        draw.text((20,12),'CliffWalking | Trained Q-learning agent | No exploration',font=font,fill='black')
        draw.text((20,46),f'Seed 0   |   Step {step}   |   Return {total:.0f}   |   Action noise p = 0',font=font,fill='black')
        payload=np.asarray(img).tobytes()
        for _ in range(repeat):pipe.stdin.write(payload)
    add(frame,12)
    for step in range(1,best.max_steps+1):
        state,reward,terminated,truncated,_=env.step(int(np.argmax(q[state])))
        total+=reward;add(env.render(),4)
        if terminated or truncated:break
    add(env.render(),16)
    pipe.stdin.close()
    if pipe.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
    env.close()
    assert (step,total,state)==(13,-13,47)
    (out.parent/'video_metadata.json').write_text(json.dumps(dict(config=best.name,seed=0,
        evaluation_seed=300000,steps=step,return_value=total,success=True,fps=8,
        source='Gymnasium rgb_array rendering'),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results')
    parser.add_argument('--skip-video',action='store_true');args=parser.parse_args()
    root=Path(args.results);figs=root.parent/'plots';figs.mkdir(exist_ok=True)
    set_style()
    training=pd.read_csv(root/'training.csv');checkpoints=pd.read_csv(root/'checkpoints.csv')
    early=pd.read_csv(root/'early_checkpoints.csv');test=pd.read_csv(root/'test.csv')
    robustness=pd.read_csv(root/'robustness.csv')
    summary,rob=metrics(root,training,checkpoints,early,test,robustness)
    grid_figure(figs/'01_environment.png')
    learning_figure(training,early,['baseline'],['Baseline'],figs/'02_baseline.png','Baseline')
    learning_figure(training,early,['alpha_0.1','baseline','alpha_0.9'],
                    ['α = 0.1','α = 0.5','α = 0.9'],figs/'03_alpha.png','Learning rate')
    learning_figure(training,early,['gamma_0.5','gamma_0.9','baseline'],
                    ['γ = 0.50','γ = 0.90','γ = 0.99'],figs/'04_gamma.png','Discount factor')
    learning_figure(training,early,['decay_0.95','baseline','decay_0.999'],
                    ['Decay = 0.950','Decay = 0.995','Decay = 0.999'],figs/'05_decay.png','Exploration decay')
    route_figure(figs/'06_best_route.png')
    best=json.loads((root/'best_config.json').read_text())['name']
    fig,axs=plt.subplots(1,2,figsize=(6.65,3.1),layout='constrained')
    for n,label,c in zip([best,'noise_0.10','noise_0.25'],['p = 0.00','p = 0.10','p = 0.25'],COLORS):
        x,m,s=smooth_curve(training,n);band(axs[0],x,m,s,c,label)
        d=checkpoints[checkpoints.config==n].pivot(index='episode',columns='train_seed',values='reward')
        band(axs[1],d.index.to_numpy(),d.mean(axis=1).to_numpy(),d.std(axis=1).to_numpy(),c,label)
    axs[0].set(xlabel='Training episode',ylabel='100-episode mean return',title='Training under action noise')
    axs[1].set(xlabel='Training episode',ylabel='Greedy validation mean return',title='Policy stability under noise')
    for ax in axs:ax.legend(frameon=False,loc='lower right')
    fig.savefig(figs/'07_noise_learning.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(6.65,3.25),layout='constrained')
    x=np.arange(3);width=.34
    for i,phase in enumerate(['zero_noise_transfer','matched_noise']):
        d=rob[rob.phase==phase].set_index('noise')
        if phase=='matched_noise':
            zero=rob[(rob.phase=='zero_noise_transfer')&(rob.noise==0)].copy()
            zero['phase']=phase;d=pd.concat([zero.set_index('noise'),d])
        d=d.reindex([0,.1,.25])
        label='Trained at p = 0' if i==0 else 'Trained at test noise'
        for ax,col,mult in [(axs[0],'reward',1),(axs[1],'success',100)]:
            ax.bar(x+(i-.5)*width,d[col+'_mean']*mult,width,
                   yerr=d[col+'_seed_sd']*mult,capsize=3,color=COLORS[i],label=label,
                   error_kw=dict(lw=.8))
    axs[0].set(ylabel='Mean test return',title='Reward under the same test noise')
    axs[1].set(ylabel='Goal reached (%)',title='Success within 500 steps',ylim=(0,112))
    for ax in axs:
        ax.set_xticks(x,['0.00','0.10','0.25']);ax.set_xlabel('Action replacement probability p')
        ax.grid(axis='y',alpha=.18)
    axs[0].legend(frameon=False,fontsize=8.3,loc='lower left')
    fig.savefig(figs/'08_robustness.png',bbox_inches='tight');plt.close(fig)
    if not args.skip_video:record_video(root,root.parent/'videos/trained_agent.mp4')
    print(summary.to_string(index=False))
    print(rob.to_string(index=False))


if __name__=='__main__':main()
