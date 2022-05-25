from collections import Counter

from torchtext.vocab import pretrained_aliases, Vocab, Vectors
from tqdm import tqdm


def build_vocab_from_iterator(iterator, emb_dim, voc_size, min_freq, path_to_vectors):
    """
    Build a Vocab from an iterator.

    Arguments:d
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
    """

    counter = Counter()
    with tqdm(unit='lines', desc='Building Vocabulary') as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    if emb_dim in pretrained_aliases:
        word_vocab = Vocab(counter, max_size=voc_size, min_freq=min_freq, vectors=emb_dim, vectors_cache=path_to_vectors,
                           specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    else:
        custom_vectors = Vectors(emb_dim, path_to_vectors)
        word_vocab = Vocab(counter, max_size=voc_size, min_freq=min_freq, vectors=custom_vectors, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    return word_vocab