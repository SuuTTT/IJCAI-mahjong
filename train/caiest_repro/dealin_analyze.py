import numpy as np, time
t0=time.time()
d=np.load('/root/final2_harvest/final2_cai_corpus.npz', allow_pickle=True)
obs=d['obs']; act=d['act'].astype(np.int64); seat=d['seat'].astype(np.int64)
kind=d['kind'].astype(np.int64); score=d['score'].astype(np.int64); fan=d['fan'].astype(np.int64)
game=d['game'].astype(np.int64); step=d['step'].astype(np.int64); gamelen=d['gamelen'].astype(np.int64)
N=len(act)
PLAY,CHI=2,36
is_disc=(act>=PLAY)&(act<CHI)
print("N rows=%d  discard rows=%d (%.3f)"%(N, is_disc.sum(), is_disc.mean()))
print("kind counts:", np.bincount(kind))
print("fan distribution (nonzero):", np.unique(fan[fan>0])[:20], " n fan>0 rows=",(fan>0).sum())

# unique games
ug=np.unique(game)
print("unique games=%d  mean gamelen=%.1f"%(len(ug), gamelen[np.unique(game,return_index=True)[1]].mean()))

# Build per-(game,seat) score & fan. Verify score constant per (game,seat)
# Use dict via sort
order=np.lexsort((seat,game))
g_s=game[order]; se_s=seat[order]; sc_s=score[order]; fa_s=fan[order]
# group by game
ron_games=0; zimo_games=0; draw_games=0; ambig=0
pos_rows=0
labels=np.zeros(N, dtype=np.int8)
# index rows by game for labeling
gi_sort=np.argsort(game, kind='stable')
gsorted=game[gi_sort]
# boundaries
bnd=np.searchsorted(gsorted, ug)
bnd=np.append(bnd, N)
score_const_ok=True
for gi,gid in enumerate(ug):
    rows=gi_sort[bnd[gi]:bnd[gi+1]]
    ss=seat[rows]; scc=score[rows]; faa=fan[rows]; stt=step[rows]; aa=act[rows]
    # per seat score
    seats=np.unique(ss)
    seat_score={}; seat_fan={}
    for s in seats:
        m=ss==s
        vs=np.unique(scc[m])
        if len(vs)!=1: score_const_ok=False
        seat_score[s]=scc[m][0]
        seat_fan[s]=faa[m].max()
    winners=[s for s in seats if seat_fan[s]>0]
    if len(winners)==0:
        draw_games+=1; continue
    w=winners[0]
    # losers among all 4 seats present
    loser_scores={s:seat_score[s] for s in seats if s!=w}
    if not loser_scores:
        continue
    mn=min(loser_scores.values())
    mins=[s for s,v in loser_scores.items() if v==mn]
    # ron if unique strict min AND min< the others (i.e. more than one distinct loser value)
    distinct=set(loser_scores.values())
    if len(mins)==1 and len(distinct)>1:
        ron_games+=1
        disc=mins[0]
        # discarder's last discard row
        dm=(ss==disc)&(aa>=PLAY)&(aa<CHI)
        if dm.any():
            drows=rows[dm]; dsteps=stt[dm]
            lr=drows[np.argmax(dsteps)]
            labels[lr]=1; pos_rows+=1
        else:
            ambig+=1
    else:
        zimo_games+=1
print("ron=%d zimo=%d draw=%d ambig(no discard row for discarder)=%d"%(ron_games,zimo_games,draw_games,ambig))
print("score constant per (game,seat):", score_const_ok)
print("POSITIVE discard rows=%d"%pos_rows)
print("per-DISCARD positive rate = %.4f"%(pos_rows/is_disc.sum()))
# per-(seat,game) deal-in rate
tot_seat_games=sum(len(np.unique(seat[gi_sort[bnd[gi]:bnd[gi+1]]])) for gi in range(len(ug)))
print("per-(seat,game) deal-in rate = %.4f  (ron_games/tot_seat_games=%d/%d)"%(ron_games/tot_seat_games, ron_games, tot_seat_games))
np.save('/tmp/dealin_labels.npy', labels)
print("saved labels; elapsed %.1fs"%(time.time()-t0))
