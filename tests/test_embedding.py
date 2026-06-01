from orbit_or.embedding import EMBEDDING_DIM, get_embedding, get_embeddings_batch


def test_single_embedding():
    text = "This is a test sentence."
    embedding = get_embedding(text)

    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in embedding)


def test_batch_embeddings():
    texts = ["First sentence.", "Second sentence."]
    embeddings = get_embeddings_batch(texts)

    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == EMBEDDING_DIM
    assert len(embeddings[1]) == EMBEDDING_DIM


def test_empty_string():
    assert get_embedding("") is None
    assert get_embedding("   ") is None
