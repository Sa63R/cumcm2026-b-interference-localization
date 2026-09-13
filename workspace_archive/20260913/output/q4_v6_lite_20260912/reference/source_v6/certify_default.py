"""Recompute the continuous fixed-layout certificate used by all versions."""
import argparse,json,math,pathlib
from q4_coverage import certify
from q4_v4_solver import V4Config

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='coverage_certificate.json');a=p.parse_args()
    n1,n2,r1,r2=V4Config().ring_sites
    sites=[(0.,0.)]+[(r*math.cos(2*math.pi*k/n),r*math.sin(2*math.pi*k/n))for n,r in [(n1,r1),(n2,r2)]for k in range(n)]
    result=certify(sites,keep_leaves=True)
    if not result['ok']:raise RuntimeError('Continuous certificate failed')
    result['sites']=sites;result['note']='Conservative floating-point continuous-box certificate; not a formally verified interval-arithmetic proof. Every accepted box is enclosed by nearby station hull; unresolved boxes never authorize omission.'
    pathlib.Path(a.out).write_text(json.dumps(result,ensure_ascii=False));print({k:v for k,v in result.items()if k not in ['sites','leaves']})
if __name__=='__main__':main()
