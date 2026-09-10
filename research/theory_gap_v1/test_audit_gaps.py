import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('gap_audit', Path(__file__).with_name('audit_gaps.py'))
gap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gap)


def test_mean_of_ratios_is_not_ratio_of_means():
    rows = [dict(virtual_time_s=t, original_lower_s=b, cited_six_disk_lower_s=b,
                 successful=True, failed_clear_count=0, seed=i,
                 movement_s=t, switching_s=0, detection_s=0, optical_s=0, removal_s=0)
            for i,(t,b) in enumerate(((200.,100.), (1100.,1000.)))]
    result = gap.statistics_for(rows)
    assert result['original']['per_case_ratio']['mean'] == pytest.approx(1.55)
    assert result['original']['ratio_of_means'] == pytest.approx(1300/1100)
    assert result['original']['per_case_ratio']['p95'] == pytest.approx(1.955)


@pytest.mark.parametrize('change', ['protocol', 'missing', 'out_of_scope'])
def test_only_exact_common_validation_manifest_is_accepted(change):
    manifest = dict(split='validation', seeds=list(range(6000,6048)),
                    identity=dict(protocol_sha256=gap.PROTOCOL_SHA256))
    gap.validate_seed_manifest(manifest)
    if change == 'protocol':
        manifest['identity']['protocol_sha256'] = 'altered'
    elif change == 'missing':
        manifest['seeds'].pop()
    else:
        manifest['split'] = 'final_random'
    with pytest.raises(ValueError, match='complete frozen common validation'):
        gap.validate_seed_manifest(manifest)


def test_changed_external_helper_is_refused_before_import(tmp_path):
    (tmp_path/'audit_eval_bounds.py').write_text("raise AssertionError('Do not execute')", encoding='utf-8')
    with pytest.raises(ValueError, match='External theory helper changed'):
        gap.helpers(tmp_path)
