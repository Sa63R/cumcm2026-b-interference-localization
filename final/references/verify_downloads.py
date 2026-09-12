"""Verify personal reading copies and record their source URLs and hashes."""
from pathlib import Path
import hashlib
import json

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "cumcm2018_B217_rgv.pdf": "https://dxs.moe.gov.cn/zx/2018/1101/1541052361277.pdf",
    "cumcm2018_A440_thermal_design.pdf": "https://dxs.moe.gov.cn/zx/2018/1101/1541052741358.pdf",
    "cumcm2023_official_awards.pdf": "https://www.mcm.edu.cn/upload_cn/node/701/6XE4ZF5Oc3573e0779f6cd8e31d79a6e9f6fd13d.pdf",
    "hart1968_astar.pdf": "https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf",
    "schulman2017_ppo.pdf": "https://arxiv.org/pdf/1707.06347",
    "schulman2016_gae.pdf": "https://arxiv.org/pdf/1506.02438",
    "agarwal2021_statistical_precipice.pdf": "https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf",
    "kool2019_attention_routing.pdf": "https://arxiv.org/pdf/1803.08475",
    "kassis2026_scientific_agent_skills.pdf": "https://arxiv.org/pdf/2609.00065",
}

records = []
for filename, url in SOURCES.items():
    path = ROOT / filename
    data = path.read_bytes()
    assert data.startswith(b"%PDF-"), filename
    reader = PdfReader(path)
    assert len(reader.pages) > 0, filename
    records.append({
        "filename": filename,
        "url": url,
        "bytes": len(data),
        "pages": len(reader.pages),
        "sha256": hashlib.sha256(data).hexdigest(),
    })

manifest = {
    "verified_on": "2026-09-12",
    "purpose": "Local reading copies; source URLs and notes are suitable for version control.",
    "files": records,
    "unavailable_full_text": [{
        "citation": "held1962sequencing",
        "url": "https://epubs.siam.org/doi/pdf/10.1137/0110015",
        "reason": "Publisher access-verification response; bibliographic record was verified, full text was not downloaded.",
    }],
}
(ROOT / "download_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({"verified": len(records), "pages": {r['filename']: r['pages'] for r in records}}, ensure_ascii=False))
