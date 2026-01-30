from src.rag.chain import rag_chain

def test_rag_streaming():
    gen = rag_chain("What is diabetes?")

    token = None
    for token in gen:
        break

    assert token is None or isinstance(token, str)