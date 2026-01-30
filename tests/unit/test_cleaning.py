from pathlib import Path

def test_chunks_created():
    assert Path("data/processed/chunks").exists()
