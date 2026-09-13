"""Conditional Monte Carlo reference for four previously observed official Q3 maps.
No official calls, strategy execution, simulated measurement or old scenario data.
"""
import bisect
import hashlib
import itertools
import json
import math
import pathlib
import random
import statistics
import time

SOURCE = pathlib.Path('/tmp/cumcm_official_q3_observation_inventory.json')
DESTINATION = pathlib.Path('/tmp/cumcm_official_q3_distribution_reference.json')
RADIUS = 1800.0
REPLICATES = 10000
MC_SEED = 4720260912
METRICS = ('mean_normalized_radius_squared', 'radial_uniform_ks', 'minimum_pair_distance_m', 'mean_nearest_neighbor_distance_m')


def weighted_ks_parts(values, weights):
    cumulative = plus = minus = 0.0
    for value, weight in sorted(zip(values, weights)):
        minus = max(minus, value-cumulative)
        cumulative += weight
        plus = max(plus, cumulative-value)
    return plus, minus


def weighted_ks(values, weights):
    return max(weighted_ks_parts(values, weights))


def metrics(points):
    n = len(points)
    u = [(x*x+y*y)/(RADIUS*RADIUS) for x,y in points]
    nearest = [math.inf]*n
    for i in range(n):
        for j in range(i):
            d = math.dist(points[i], points[j])
            nearest[i] = min(nearest[i],d)
            nearest[j] = min(nearest[j],d)
    return dict(zip(METRICS, (statistics.fmean(u), weighted_ks(u,[1/n]*n), min(nearest), statistics.fmean(nearest))))


def observed_world(world):
    sources = world['source_estimates']
    n = len(sources)
    points = [source['center'] for source in sources]
    estimates = metrics(points)
    rows = []
    lows, highs = [], []
    for source in sources:
        radial = math.hypot(*source['center'])
        lower = max(0., radial-source['enclosure_radius_m'])
        upper = min(RADIUS, radial+source['enclosure_radius_m'])
        low, high = (lower/RADIUS)**2, (upper/RADIUS)**2
        lows.append(low); highs.append(high)
        rows.append({'channel':source['channel'], 'estimated_radius_m':radial,
                     'radius_interval_m':[lower,upper],
                     'normalized_radius_squared':(radial/RADIUS)**2,
                     'normalized_radius_squared_interval':[low,high],
                     'position_enclosure_radius_m':source['enclosure_radius_m']})
    nn_low, nn_high = [math.inf]*n, [math.inf]*n
    distances = []
    for i in range(n):
        for j in range(i):
            d = math.dist(points[i],points[j])
            uncertainty = sources[i]['enclosure_radius_m']+sources[j]['enclosure_radius_m']
            low, high = max(0.,d-uncertainty),d+uncertainty
            distances.append({'channels':[sources[i]['channel'],sources[j]['channel']],
                              'distance_estimate_m':d,'conservative_interval_m':[low,high]})
            nn_low[i]=min(nn_low[i],low); nn_low[j]=min(nn_low[j],low)
            nn_high[i]=min(nn_high[i],high); nn_high[j]=min(nn_high[j],high)
    plus_low,minus_high = weighted_ks_parts(highs,[1/n]*n)
    plus_high,minus_low = weighted_ks_parts(lows,[1/n]*n)
    intervals = {'mean_normalized_radius_squared':[statistics.fmean(lows),statistics.fmean(highs)],
                 'radial_uniform_ks':[max(0.,plus_low,minus_low),max(plus_high,minus_high)],
                 'minimum_pair_distance_m':[min(nn_low),min(nn_high)],
                 'mean_nearest_neighbor_distance_m':[statistics.fmean(nn_low),statistics.fmean(nn_high)]}
    for metric in METRICS:
        assert intervals[metric][0]-1e-12 <= estimates[metric] <= intervals[metric][1]+1e-12
    return {'case_id':world['case'],'source_count':n,'estimates':estimates,
            'conservative_statistic_intervals':intervals,'radial_source_records':rows,
            'closest_estimated_pair':min(distances,key=lambda row:row['distance_estimate_m']),
            'nearest_neighbor_records':[{'channel':s['channel'],'distance_interval_m':[a,b]}
                                        for s,a,b in zip(sources,nn_low,nn_high)]}


def quantile(values,q):
    t=q*(len(values)-1); lo=int(t); hi=math.ceil(t)
    return values[lo]+(values[hi]-values[lo])*(t-lo)


def probabilities(samples,value):
    n=len(samples)
    lower=(bisect.bisect_right(samples,value)+1)/(n+1)
    upper=(n-bisect.bisect_left(samples,value)+1)/(n+1)
    return lower,upper,min(1.,2*min(lower,upper))


def reference(samples,estimate,bounds,metric):
    samples=sorted(samples)
    plow,pup,ptwo=probabilities(samples,estimate)
    low,high=bounds
    if metric=='radial_uniform_ks':
        alternative='larger discrepancy from area-uniform radial CDF'
        pvalue=pup
        prange=[probabilities(samples,high)[1],probabilities(samples,low)[1]]
    else:
        alternative=('two-sided radial deviation; lower suggests more central, higher more peripheral'
                     if metric=='mean_normalized_radius_squared' else
                     'two-sided spacing deviation; lower suggests clustering, higher separation')
        pvalue=ptwo
        endpoints=[probabilities(samples,v)[2] for v in (low,high)]
        maximum=1. if low<=statistics.median(samples)<=high else max(endpoints)
        prange=[min(endpoints),maximum]
    return {'observed_estimate':estimate,'observed_conservative_interval':bounds,
            'null_mean':statistics.fmean(samples),'null_median':statistics.median(samples),
            'null_central_95_percent_interval':[quantile(samples,.025),quantile(samples,.975)],
            'lower_tail_probability_at_estimate':plow,'upper_tail_probability_at_estimate':pup,
            'test_alternative':alternative,'exploratory_p_at_estimate':pvalue,
            'p_range_allowed_by_position_uncertainty':prange,
            'conservative_exploratory_p':prange[1]}


def main():
    began=time.perf_counter()
    raw=SOURCE.read_bytes(); data=json.loads(raw)
    observed=[observed_world(world) for world in data['worlds']]
    n_maps=len(observed)
    distributions=[{m:[] for m in METRICS} for _ in observed]
    aggregate_samples={m:[] for m in METRICS}
    generator=random.Random(MC_SEED)
    for repeat in range(REPLICATES):
        group=[]; combined_u=[]; combined_weights=[]
        for i,world in enumerate(observed):
            n=world['source_count']
            points=[]
            for j in range(n):
                radial=RADIUS*math.sqrt(generator.random())
                angle=2*math.pi*generator.random()
                points.append((radial*math.cos(angle),radial*math.sin(angle)))
            stats=metrics(points);group.append(stats)
            for m in METRICS:distributions[i][m].append(stats[m])
            combined_u.extend((x*x+y*y)/(RADIUS*RADIUS) for x,y in points)
            combined_weights.extend([1/(n_maps*n)]*n)
        for m in METRICS:
            value=(weighted_ks(combined_u,combined_weights) if m=='radial_uniform_ks'
                   else statistics.fmean(row[m] for row in group))
            aggregate_samples[m].append(value)
    for world,reference_samples in zip(observed,distributions):
        world['monte_carlo_references']={m:reference(reference_samples[m],world['estimates'][m],
            world['conservative_statistic_intervals'][m],m) for m in METRICS}
    weights=[];u=[];low_u=[];high_u=[]
    for world in observed:
        n=world['source_count'];weights.extend([1/(n_maps*n)]*n)
        for source in world['radial_source_records']:
            u.append(source['normalized_radius_squared'])
            low_u.append(source['normalized_radius_squared_interval'][0])
            high_u.append(source['normalized_radius_squared_interval'][1])
    overall_estimates={m:statistics.fmean(world['estimates'][m] for world in observed) for m in METRICS}
    overall_estimates['radial_uniform_ks']=weighted_ks(u,weights)
    overall_bounds={m:[statistics.fmean(world['conservative_statistic_intervals'][m][i] for world in observed)
                       for i in (0,1)] for m in METRICS}
    lower_plus,upper_minus=weighted_ks_parts(high_u,weights)
    upper_plus,lower_minus=weighted_ks_parts(low_u,weights)
    overall_bounds['radial_uniform_ks']=[max(0.,lower_plus,lower_minus),max(upper_plus,upper_minus)]
    overall={m:reference(aggregate_samples[m],overall_estimates[m],overall_bounds[m],m) for m in METRICS}
    all_tests=[]
    for world in observed:
        all_tests.extend((world['case_id'],m,x) for m,x in world['monte_carlo_references'].items())
    all_tests.extend(('four_maps_equal_weight',m,x) for m,x in overall.items())
    correction=0.
    for index,(case,m,row) in enumerate(sorted(all_tests,key=lambda item:item[2]['conservative_exploratory_p'])):
        correction=min(1.,max(correction,(len(all_tests)-index)*row['conservative_exploratory_p']))
        row['holm_adjusted_conservative_p_across_all_explorations']=correction
    result={'analysis_kind':'conditional_iid_area_uniform_disk_reference_for_four_official_observation_maps',
            'source_inventory':str(SOURCE),'source_inventory_sha256':hashlib.sha256(raw).hexdigest(),
            'script':str(pathlib.Path(__file__).resolve()),
            'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            'maps':len(observed),'source_count':sum(x['source_count'] for x in observed),
            'arena_radius_m':RADIUS,'conditional_source_counts':[x['source_count'] for x in observed],
            'null_hypothesis':'Within each map, locations are independent and area-uniform in the radius-1800 disk; maps are independent; each observed N is held fixed.',
            'null_radial_law':'u=(r/1800)^2 is Uniform(0,1); radius r itself is not uniform.',
            'null_generation':'r=1800*sqrt(U), theta=2*pi*V; U,V from a local statistical-reference RNG only.',
            'monte_carlo_replicates':REPLICATES,'monte_carlo_seed':MC_SEED,
            'monte_carlo_seed_scope':'Statistical reference seed, not a simulator, official-case or strategy seed.',
            'cluster_unit':'One full map. Overall statistics give each of the four maps equal weight. Pair distances are never treated as independent observations.',
            'overall_radial_ks_definition':'Maximum absolute discrepancy of equal-map-weight empirical radial CDF from Uniform(0,1).',
            'geometric_uncertainty':'Each estimated source lies in its recorded enclosing circle under the public stationary-source and measurement-error model. All intervals use triangle inequalities; interval propagation may be loose.',
            'ks_uncertainty':'Bound the true CDF between the CDF of upper radial endpoints and the CDF of lower radial endpoints; derive conservative lower/upper KS discrepancy bounds.',
            'probability_convention':'Add-one Monte Carlo tail estimates. Non-KS tests use twice the smaller empirical tail. Conservative p is the maximum over the propagated statistic interval.',
            'multiple_exploration_tests':len(all_tests),'multiplicity_adjustment':'Holm adjustment across all four per-map metrics and four overall metrics (20 exploratory comparisons).',
            'limitations':['Only four archived official maps; sample inclusion and generator stationarity are not established.',
                           'Estimated coordinates are not hidden official truth; their conservative uncertainty is carried into metrics.',
                           'The area-uniform iid null is a comparison model, not a claim about the official generator.',
                           'These explorations are not preregistered confirmation; 10000 Monte Carlo replicates do not increase the number of observed official maps.',
                           'Failure to reject the null does not establish uniformity, independence, the absence of weak clustering, or any PRNG design.',
                           'No archived 872-case database, old sealed evaluation, official API or strategy/simulator execution was used.'],
            'worlds':observed,'overall_equal_map_weight':overall,
            'runtime_s':time.perf_counter()-began}
    DESTINATION.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(DESTINATION),'runtime_s':result['runtime_s'],
                     'worlds':[{'case_id':w['case_id'],'N':w['source_count'],
                                'references':w['monte_carlo_references']} for w in observed],
                     'overall_equal_map_weight':overall},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
