import json,numpy as np,statistics,time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from q4_transit_critic import Critic
rows=[json.loads(s)for s in open('critic_training.jsonl')]
X=np.asarray([r['features']for r in rows]);y=np.asarray([r['saving_seconds']for r in rows]);w=1/np.asarray([r['targets']for r in rows])
train=np.array([r['seed']%100<48 for r in rows])
info={'training_cases':len(set(r['seed']for r in rows)),'counterfactual_examples':len(rows),'feature_count':X.shape[1],'target':'V4 full-completion baseline seconds minus alternative seconds','labels_mean':float(y.mean()),'models':{}}
for loss in ['huber','squared_error']:
    model=GradientBoostingRegressor(loss=loss,n_estimators=180,learning_rate=.05,max_depth=3,min_samples_leaf=35,random_state=58173,subsample=.85,alpha=.85)
    model.fit(X[train],y[train],sample_weight=w[train]);pred=model.predict(X[~train])
    metrics={'heldout_MSE':float(mean_squared_error(y[~train],pred)),'zero_MSE':float(np.mean(y[~train]**2)),'heldout_corr':float(np.corrcoef(y[~train],pred)[0,1])}
    model.fit(X,y,sample_weight=w)
    d={'base':float(model.init_.constant_[0,0]),'learning_rate':model.learning_rate,'feature_count':X.shape[1],'trees':[],'training':metrics}
    for est in model.estimators_[:,0]:
        tr=est.tree_;d['trees'].append({'feature':tr.feature.tolist(),'threshold':tr.threshold.tolist(),'left':tr.children_left.tolist(),'right':tr.children_right.tolist(),'value':tr.value[:,0,0].tolist()})
    fname=f'transit_critic_{loss}.json';open(fname,'w').write(json.dumps(d))
    portable=Critic(fname);err=max(abs(portable.predict(x)-a)for x,a in zip(X[:1000],model.predict(X[:1000])))
    assert err<1e-5,err
    metrics['export_max_abs_error']=err;info['models'][loss]=metrics
    print(loss,metrics,flush=True)
open('critic_training_summary.json','w').write(json.dumps(info,indent=2))
