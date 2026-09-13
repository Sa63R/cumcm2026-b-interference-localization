import json,numpy as np,time
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from q4_transit_critic import Critic
info={}
for name,path in [('transit','critic_training.jsonl'),('route','route_critic_training.jsonl')]:
    rows=[json.loads(s)for s in open(path)];X=np.asarray([r['features']for r in rows]);y=np.asarray([r['saving_seconds']for r in rows]);w=1/np.asarray([r['targets']for r in rows]);train=np.array([r['seed']%100<48 for r in rows])
    for key,klass in [('extra',ExtraTreesRegressor),('forest',RandomForestRegressor)]:
        model=klass(n_estimators=128,max_depth=12,min_samples_leaf=15,random_state=822173,n_jobs=2,max_features=.9)
        model.fit(X[train],y[train],sample_weight=w[train]);pred=model.predict(X[~train]);metrics={'MSE':float(mean_squared_error(y[~train],pred)),'zero_MSE':float(np.mean(y[~train]**2)),'corr':float(np.corrcoef(y[~train],pred)[0,1])}
        model.fit(X,y,sample_weight=w);d={'base':0.,'learning_rate':1/len(model.estimators_),'feature_count':X.shape[1],'trees':[],'training':metrics}
        for est in model.estimators_:
            tr=est.tree_;d['trees'].append({'feature':tr.feature.tolist(),'threshold':tr.threshold.tolist(),'left':tr.children_left.tolist(),'right':tr.children_right.tolist(),'value':tr.value[:,0,0].tolist()})
        fn=f'{name}_critic_{key}.json';open(fn,'w').write(json.dumps(d));portable=Critic(fn)
        err=max(abs(portable.predict(x)-a)for x,a in zip(X[:1000],model.predict(X[:1000])));assert err<1e-5
        metrics['export_error']=err;info[name+'_'+key]=metrics;print(name,key,metrics,flush=True)
open('extra_training_summary.json','w').write(json.dumps(info,indent=2))
