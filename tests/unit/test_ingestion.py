from pathlib import Path

def test_processed_data_exists():
    assert Path("data/processed").exists()
