"""Residual LightGBM gain for the user-kNN scorer."""
from .graph_residual_per_scorer import *

def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz'))
    B=D0+4*DAY; bounds=bounds_for(4,half=True)
    train,val,_=train_frames(data,B,bounds,'a3_c1_c2',choice_maxiter=20)
    parts=[];off=0
    for wi,b in enumerate(bounds):
        n=int((train['window_id'].to_numpy()==wi).sum()); parts.append(add_base(train.iloc[off:off+n].copy(),np.load(OUTPUT_DIR/f'logmf_cosine_prod{wi}_{int(b)}.npy'))); off+=n
    train=pd.concat(parts,ignore_index=True); val=add_base(val,np.load(OUTPUT_DIR/f'logmf_cosine_val_dm24_{int(B)}.npy')); evalf=cached_frame(data,CUT,'eval')
    base,_=fit_one(train,val,evalf)
    trparts=[];off=0
    for wi,b in enumerate(bounds):
        n=int((train['window_id'].to_numpy()==wi).sum()); part=train.iloc[off:off+n].copy(); z=np.load(OUTPUT_DIR/f'graph_scores_{int(b)}.npz')['user_knn']; q=z.reshape(-1,10); part['g_user_knn_score']=z; part['g_user_knn_rank']=np.argsort(np.argsort(-q,axis=1),axis=1).astype(np.float32).reshape(-1); part['g_user_knn_gap']=(q-q.max(axis=1,keepdims=True)).astype(np.float32).reshape(-1); trparts.append(part); off+=n
    tr=pd.concat(trparts,ignore_index=True); z=np.load(OUTPUT_DIR/f'graph_scores_{int(B)}.npz')['user_knn']; q=z.reshape(-1,10); va=val.copy(); va['g_user_knn_score']=z; va['g_user_knn_rank']=np.argsort(np.argsort(-q,axis=1),axis=1).astype(np.float32).reshape(-1); va['g_user_knn_gap']=(q-q.max(axis=1,keepdims=True)).astype(np.float32).reshape(-1)
    dm,it=fit_one(tr,va,evalf); res={'base_dm24':base,'user_knn_dm24':dm,'gain':dm-base,'iteration':it}; (OUTPUT_DIR/'graph_user_residual.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
