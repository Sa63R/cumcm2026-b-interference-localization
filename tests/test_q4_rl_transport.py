"""Object listing integrity; no credentials, SDK or network used."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("q4_transport", Path(__file__).resolve().parents[1] / "scripts/q4_object_exchange.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ListingClient:
    def __init__(self, pages):
        self.pages, self.requests = list(pages), []
    def list_objects_v2(self, **request):
        self.requests.append(request)
        return self.pages[len(self.requests) - 1]


def test_all_three_pages_are_returned_beyond_original_200_limit():
    prefix = "task/"
    pages = []
    for start, stop, token in ((0, 200, "page2"), (200, 400, "page3"), (400, 407, None)):
        page = {"Contents": [{"Key": f"task/run/{i:04d}.json", "Size": i} for i in range(start, stop)],
                "IsTruncated": token is not None}
        if token:
            page["NextContinuationToken"] = token
        pages.append(page)
    client = ListingClient(pages)
    result = module.list_all_objects(client, "bucket", prefix, "run/")
    assert len(result) == 407
    assert result[-1] == {"name": "run/0406.json", "bytes": 406}
    assert "ContinuationToken" not in client.requests[0]
    assert [r["ContinuationToken"] for r in client.requests[1:]] == ["page2", "page3"]


@pytest.mark.parametrize("pages", [
    [{"IsTruncated": True}],
    [{"IsTruncated": True, "NextContinuationToken": "repeat"},
     {"IsTruncated": True, "NextContinuationToken": "repeat"}],
])
def test_broken_truncated_cursor_fails_instead_of_returning_partial_list(pages):
    with pytest.raises(ValueError, match="continuation"):
        module.list_all_objects(ListingClient(pages), "bucket", "task/")


def test_cross_prefix_response_is_rejected():
    client = ListingClient([{"Contents": [{"Key": "other/private", "Size": 1}]}])
    with pytest.raises(ValueError, match="escaped"):
        module.list_all_objects(client, "bucket", "task/")
