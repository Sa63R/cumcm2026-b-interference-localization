import json,numpy as np,os,time,warnings
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from q4_transit_critic import Critic
warnings.filterwarnings('ignore',message='X does not have valid feature names')
rows=[json.loads(s)for p in ['critic_training.jsonl','critic_training_extra.jsonl']for s in open(p)]
X=np.asarray([r['features']for r in rows],dtype=np.float32);y=np.asarray([r['saving_seconds']for r in rows]);w=1/np.asarray([r['targets']for r in rows]);train=np.array([r['seed']%10<8 for r in rows]);info={'training_cases':len(set(r['seed']for r in rows)),'examples':len(rows),'models':{}}
for key in ['light','extra','huber']:
 if key=='light':model=LGBMRegressor(objective='regression',n_estimators=280,learning_rate=.04,num_leaves=20,max_depth=6,min_child_samples=60,random_state=822174,n_jobs=2,verbosity=-1,reg_lambda=5.)
 elif key=='extra':model=ExtraTreesRegressor(n_estimators=160,max_depth=15,min_samples_leaf=25,random_state=822174,n_jobs=2,max_features=.9)
 else:model=GradientBoostingRegressor(loss='huber',alpha=.85,n_estimators=220,learning_rate=.05,max_depth=3,min_samples_leaf=60,random_state=822174,subsample=.85)
 model.fit(X[train],y[train],sample_weight=w[train]);pred=model.predict(X[~train]);metrics={'MSE':float(mean_squared_error(y[~train],pred)),'zero_MSE':float(np.mean(y[~train]**2)),'corr':float(np.corrcoef(y[~train],pred)[0,1])}
 model.fit(X,y,sample_weight=w)
 d={'base':0.,'learning_rate':1.,'feature_count':X.shape[1],'trees':[],'training':metrics,'training_cases':info['training_cases']}
 if key=='light':
  for tr in model.booster_.dump_model()['tree_info']:
   tree={'feature':[],'threshold':[],'left':[],'right':[],'value':[]}
   def visit(node):
    j=len(tree['feature']);tree['feature'].append(-1);tree['threshold'].append(0.);tree['left'].append(-1);tree['right'].append(-1);tree['value'].append(0.)
    if 'leaf_value'in node:tree['value'][j]=node['leaf_value']
    else:
     assert node['decision_type']=='<='
     tree['feature'][j]=node['split_feature'];tree['threshold'][j]=float(node['threshold']);tree['left'][j]=visit(node['left_child']);tree['right'][j]=visit(node['right_child'])
    return j
   visit(tr['tree_structure']);d['trees'].append(tree)
 else:
  if key=='extra':trees=model.estimators_;d['learning_rate']=1/len(trees)
  else:trees=model.estimators_[:,0];d['base']=float(model.init_.constant_[0,0]);d['learning_rate']=model.learning_rate
  for est in trees:
   tr=est.tree_;d['trees'].append({'feature':tr.feature.tolist(),'threshold':tr.threshold.tolist(),'left':tr.children_left.tolist(),'right':tr.children_right.tolist(),'value':tr.value[:,0,0].tolist()})
 fn='transit_critic_big_'+key+'.json';open(fn,'w').write(json.dumps(d));p=Critic(fn);metrics['export_error']=max(abs(p.predict(x)-a)for x,a in zip(X[:1000],model.predict(X[:1000])));assert metrics['export_error']<1e-5
 info['models'][key]=metrics;print(key,metrics,flush=True)
open('big_transit_training_summary.json','w').write(json.dumps(info,indent=2))
