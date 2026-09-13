"""Inference demonstration, NOT an end-to-end optimization benchmark."""
import json
from radius_direction import marginalize, visibility_probability, count_and_subset_sampler

def main():
    example = {
        'scope': 'Conditional on one candidate position, explicit planning priors, not official data',
        'candidate_position': [0, 0],
        'positive_points': [[500, 0]],
        'negative_points': [[1200, 0]],
        'probe': [1300, 0],
        'new_visibility_probability': visibility_probability((0,0),[(500,0)],[(1200,0)],(1300,0)),
        'directional_probability_after_one_positive': marginalize((0,0),[(500,0)]).posterior_directional,
        'opposite_side_visibility_after_one_positive': visibility_probability((0,0),[(500,0)],[],(-500,0)),
        'count_prior_recovery': count_and_subset_sampler([1.]*20,0)[0],
        'warning': 'A zero probability is not a certificate of nonexistence. No speedup is claimed.'
    }
    print(json.dumps(example, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
