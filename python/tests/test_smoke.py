"""スモークテスト: 各プロセスのスタブ応答が正しい形式で返ることを確認。"""

from mortal.main import handle_infer
from mortal.mortal_engine import MortalEngine
from mortal.snapshot_to_mjai import SnapshotToMjaiConverter
from recognition.main import stub_tenhou_json


def test_stub_tenhou_json_shape():
    j = stub_tenhou_json()
    assert "hand" in j
    assert "self_wind" in j
    assert isinstance(j["hand"], list)


def test_stub_inference_shape():
    """issue #20: ダミー mjai event を渡すと整形済み InferenceResult が返る。"""
    engine = MortalEngine.stub()
    r = engine.infer([{"type": "tsumo", "actor": 0, "pai": "5m"}])
    assert "recommended" in r
    assert "candidates" in r
    assert "primary_label" in r
    assert isinstance(r["candidates"], list)
    assert len(r["candidates"]) == 5
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


def test_handle_infer_end_to_end():
    """issue #20: handle_infer (snapshot → converter → engine) パイプラインのスモーク。

    recognition プロセスが emit する tenhou_json shape (= stub_tenhou_json) を
    そのまま投入し、Rust が deserialize できる shape で result が返ることを確認。
    """
    engine = MortalEngine.stub()
    converter = SnapshotToMjaiConverter()
    req = {
        "type": "infer",
        "id": 42,
        "tenhou_json": stub_tenhou_json(),
    }
    result = handle_infer(engine, req, converter=converter)
    assert result["type"] == "result"
    assert result["id"] == 42
    assert "recommended" in result
    assert len(result["candidates"]) == 5
    assert "primary_label" in result
    # Rust 側 (monitor.rs:556) は recommended / candidates / timestamp / primary_label を読む。
    assert isinstance(result["timestamp"], str)
    assert isinstance(result["primary_label"], str) and result["primary_label"]


def test_handle_infer_not_implemented_returns_error_response():
    """CodeRabbit on PR #52: `_infer_real` の NotImplementedError がプロセスを
    殺さず error 応答に変換されること。Phase D5 未実装期間にリアルモードで動かす
    と毎回再起動ループに陥るのを防ぐ重要な契約。
    """
    engine = MortalEngine()  # stub=False のまま load しないので _infer_real が呼ばれる
    engine._ready = True  # type: ignore[attr-defined]  # load() を skip して real path を直接叩く
    converter = SnapshotToMjaiConverter()
    req = {"type": "infer", "id": 7, "tenhou_json": stub_tenhou_json()}
    result = handle_infer(engine, req, converter=converter)
    assert result["type"] == "error"
    assert result["id"] == 7
    assert "Phase D5" in result["message"]


def test_handle_infer_generic_exception_returns_error_response():
    """汎用 Exception (= converter / engine がバグ等で予期せず raise) も
    プロセスを殺さず error 応答に変換される。
    """

    class FailingEngine:
        def infer(self, _events):
            raise RuntimeError("simulated inference bug")

    converter = SnapshotToMjaiConverter()
    req = {"type": "infer", "id": 8, "tenhou_json": stub_tenhou_json()}
    result = handle_infer(FailingEngine(), req, converter=converter)  # type: ignore[arg-type]
    assert result["type"] == "error"
    assert result["id"] == 8
    assert "simulated inference bug" in result["message"]
