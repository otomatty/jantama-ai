"""スモークテスト: 各プロセスのスタブ応答が正しい形式で返ることを確認。"""

from mortal.main import stub_inference
from recognition.main import stub_tenhou_json


def test_stub_tenhou_json_shape():
    j = stub_tenhou_json()
    assert "hand" in j
    assert "self_wind" in j
    assert isinstance(j["hand"], list)


def test_stub_inference_shape():
    r = stub_inference({})
    assert "recommended" in r
    assert "candidates" in r
    assert isinstance(r["candidates"], list)
    assert r["recommended"]["action_type"] in {
        "discard",
        "riichi",
        "chi",
        "pon",
        "kan",
        "ron",
        "tsumo",
        "pass",
    }
