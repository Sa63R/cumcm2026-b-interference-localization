"""Execute the injected receipt lifecycle in Node with a stub simulator binding."""

import json
import re
import shutil
import subprocess

import pytest

from practice_control import bridge
from tests.test_practice_bridge import MockBridge, state


NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is needed for isolated JavaScript lifecycle tests")


def start_expression():
    client = MockBridge([{"active": False}, {"ok": True, "run": state()}])
    client.start_practice(3)
    return client.calls[1][0].replace(bridge._IMPORT_JS, "const {Call} = globalThis.__runtime;")


def receipt_expression(nonce):
    client = MockBridge([{"nonce": nonce, "status": "completed", "reply": {
        "ok": True, "run": state(), "error_code": "request_failed"}}])
    client._reconcile_start_receipt(nonce)
    assert len(client.calls) == 1 and client.calls[0][1] is False
    assert "Call.ByID" not in client.calls[0][0]
    return client.calls[0][0]


HARNESS = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
globalThis.window = {};
let starts = 0, resolveStart, rejectStart;
globalThis.__runtime = {Call: {ByID: async (id) => {
  if (id === 3522211836) return input.scenario === 'formal'
    ? {active:true,mode:'formal',phase:'running',case_code:'OTHER-CASE'}
    : {active:false,mode:'',phase:'',case_code:''};
  if (id !== 31672008) throw Error('Unexpected binding');
  starts++;
  return await new Promise((resolve, reject) => {resolveStart=resolve; rejectStart=reject;});
}}};
const slot = '__codexPracticeStartReceiptV1';
(async () => {
  const first = eval(input.first);
  // Attach rejection handling immediately, including the evaluation-error case.
  const firstResult = first.then(value=>({value}), error=>({error:String(error)}));
  await new Promise(resolve=>setImmediate(resolve));
  if (input.scenario === 'formal') {
    process.stdout.write(JSON.stringify({starts,first:await firstResult,receipt:window[slot]}));
    return;
  }
  const initialNonce = window[slot].nonce;
  const readBefore = eval(input.read);
  let second, sentinel;
  if (input.scenario === 'pending') second = await eval(input.second);
  if (input.scenario === 'changed_owner') {
    sentinel = {nonce:'different-owner',status:'completed',payload:{marker:'preserve'}};
    window[slot] = sentinel;
  }
  if (input.scenario === 'failed_eval') rejectStart(Error('private error body must not be cached'));
  else resolveStart({ok:true,run:{active:true,mode:'practice',problem_no:3,phase:'countdown',
    case_code:'ABCD-EFGH-IJKL-MNOP',jammers:[{x:123}],token:'not-for-projection'}});
  const result = await firstResult;
  if (input.scenario === 'failed_eval') second = await eval(input.second);
  if (input.scenario === 'success') window[slot].reply.run.sources = [{x:999}];
  const readAfter = eval(input.read);
  process.stdout.write(JSON.stringify({starts,first:result,second,initialNonce,
    receipt:window[slot],readBefore,readAfter,preservedSentinel:window[slot]===sentinel}));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""


@pytest.mark.parametrize("scenario", ["success", "pending", "changed_owner", "failed_eval", "formal"])
def test_actual_js_receipt_lifecycle_without_a_simulator(scenario):
    first = start_expression()
    nonce = re.search(r'const nonce = "([a-f0-9]{32})";', first).group(1)
    result = subprocess.run(
        [NODE, "-e", HARNESS],
        input=json.dumps({"scenario": scenario, "first": first, "second": start_expression(),
                          "read": receipt_expression(nonce)}),
        text=True, capture_output=True, timeout=10, check=True,
    )
    output = json.loads(result.stdout)
    assert output["starts"] == (0 if scenario == "formal" else 1)
    if scenario == "formal":
        assert output["first"]["value"]["guard_error"]
    elif scenario == "changed_owner":
        assert output["preservedSentinel"]
        assert output["receipt"]["nonce"] == "different-owner"
        assert output["readAfter"]["status"] == "missing_or_mismatch"
    else:
        assert output["receipt"]["nonce"] == output["initialNonce"]
        if scenario == "failed_eval":
            assert output["receipt"]["status"] == "evaluation_error"
            assert "private error body" not in json.dumps(output["receipt"])
        else:
            assert output["receipt"]["status"] == "completed"
            assert output["first"]["value"]["ok"] is True
            assert "jammers" not in json.dumps(output["receipt"])
            assert "token" not in json.dumps(output["receipt"])
            assert output["readBefore"]["status"] == "pending"
            assert output["readAfter"]["status"] == "completed"
            assert "sources" not in json.dumps(output["readAfter"])
        if scenario in {"pending", "failed_eval"}:
            assert output["second"]["guard_error"]
