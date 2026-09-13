"""Additional offline estimator trial; exported inference remains stdlib-only."""
import json,numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from q4_transit_critic import Critic
info={}
for name,path in [('transit','critic_training.jsonl'),('route','route_critic_training.jsonl')]:
 rows=[json.loads(s)for s in open(path)];X=np.asarray([r['features']for r in rows],dtype=np.float32);y=np.asarray([r['saving_seconds']for r in rows]);w=1/np.asarray([r['targets']for r in rows]);train=np.array([r['seed']%100<48 for r in rows])
 for key in ['huber','regression']:
  model=LGBMRegressor(objective=key,alpha=.85,n_estimators=240,learning_rate=.04,num_leaves=12,max_depth=5,min_child_samples=50,random_state=822173,n_jobs=2,verbosity=-1,reg_lambda=1.)
  model.fit(X[train],y[train],sample_weight=w[train]);pred=model.predict(X[~train]);metrics={'MSE':float(mean_squared_error(y[~train],pred)),'corr':float(np.corrcoef(y[~train],pred)[0,1])}
  model.fit(X,y,sample_weight=w);dm=model.booster_.dump_model();d={'base':0.,'learning_rate':1.,'feature_count':X.shape[1],'trees':[],'training':metrics}
  for treeinfo in dm['tree_info']:
   tree={'feature':[],'threshold':[],'left':[],'right':[],'value':[]}
   def visit(node):
    j=len(tree['feature']);tree['feature'].append(-1);tree['threshold'].append(0.);tree['left'].append(-1);tree['right'].append(-1);tree['value'].append(0.)
    if 'leaf_value'in node:tree['value'][j]=node['leaf_value']
    else:
     assert node['decision_type']=='<='
     tree['feature'][j]=node['split_feature'];tree['threshold'][j]=float(node['threshold']);tree['left'][j]=visit(node['left_child']);tree['right'][j]=visit(node['right_child'])
    return j
   visit(treeinfo['tree_structure']);d['trees'].append(tree)
  fn=f'{name}_critic_light_{key}.json';open(fn,'w').write(json.dumps(d));portable=Critic(fn);err=max(abs(portable.predict(x)-a)for x,a in zip(X[:1000],model.predict(X[:1000])))
  assert err<1e-5,err
  metrics['export_error']=err;info[name+'_'+key]=metrics;print(name,key,metrics,flush=True)
open('light_training_summary.json','w').write(json.dumps(info,indent=2))
