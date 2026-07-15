import os,sys
sys.path.insert(0,"/root/caiest_repro"); os.chdir("/root/caiest_repro")
import numpy as np
import s6_pimc_vcut as s6
# init worker like the pool would
s6._init_worker(dict(null=False,true_state=False,n_worlds=20,k_cutoff=6))
# counters
cnt={"value":0,"res_HU":0,"res_HUANG":0,"res_other":0,"rollouts":0}
orig_val=s6._value_score
def val_wrap(o):
    cnt["value"]+=1; return orig_val(o)
s6._value_score=val_wrap
orig_ro=s6.PIMCVSim._rollout_vcut
def ro_wrap(self,seat,tile,K):
    cnt["rollouts"]+=1
    return orig_ro(self,seat,tile,K)
s6.PIMCVSim._rollout_vcut=ro_wrap
# also wrap _loop to see return of the K-limited call: distinguish by max_turns
orig_loop=s6.Sim._loop
def loop_wrap(self,max_turns):
    r=orig_loop(self,max_turns)
    if max_turns<=10:  # this is a rollout cutoff loop
        if r=="HU": cnt["res_HU"]+=1
        elif r=="HUANG": cnt["res_HUANG"]+=1
        else: cnt["res_other"]+=1
    return r
s6.Sim._loop=loop_wrap

import time
t0=time.time()
b,s,psum,ov,dec,skip,good,bad=s6._work((0,9800000))
print("one game(4 seats) placements sum",psum,"time",round(time.time()-t0,1),"s")
print("decisions",dec,"overrides",ov,"good_worlds",good)
print("counters",cnt)
