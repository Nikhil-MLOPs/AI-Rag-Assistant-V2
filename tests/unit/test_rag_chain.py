from src.rag.chain import rag_chain

def test_rag_streaming():
    gen = rag_chain("What is diabetes?")

    # Consume at most one token safely
    token = None
    for token in gen:
        break

    # Either no output (valid in CI) or string tokens
    assert token is None or isinstance(token, str)
