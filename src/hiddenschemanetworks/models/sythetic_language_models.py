from deep_graphs.processes import node2vec_walk
from scipy.stats.distributions import poisson
from gensim.models import Word2Vec
import networkx as nx
import random
import os

def generate_word2vec(fake_corpora,corpora_name="barabasi",embeddings_size=100,data_dir="../"):
    embedding_name = "word2vec"
    corpora_size = len(fake_corpora)

    model = Word2Vec(fake_corpora, size=100, window=5, min_count=1, workers=4)
    model.save(os.path.join(data_dir,"{0}_corpora_word2vec.model".format(corpora_name)))
    embedings_path = "{0}.{1}.{2}.{3}d.txt".format(corpora_name,embedding_name,corpora_size,embeddings_size)
    embedings_path = os.path.join(data_dir,embedings_path)
    f = open(embedings_path,"w")
    for a in model.wv.vocab.keys():
        line = a+" "+" ".join(map(str,model.wv[a]))+"\n"
        f.write(line)
    f.close()

def generate_random_vocabulary(number_of_vocabulary=100,number_of_letters=3):
    all_letters_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    VOC = []
    for a in range(number_of_vocabulary):
       VOC.append("".join(random.sample(all_letters_string,number_of_letters)))
    return VOC

def generate_simple_schemata():
    """
    :return: nx graph with random words as node attributes
    """
    VOC = generate_random_vocabulary(number_of_vocabulary=1000,number_of_letters=3)
    schemata_graph = nx.barabasi_albert_graph(100,3)
    schema_to_word = {n:[] for n in schemata_graph.nodes}
    words_per_schema = 4
    for node in schemata_graph.nodes:
        node_degree = schemata_graph.degree[node]
        selected_words_per_node = random.sample(VOC,words_per_schema)
        schema_to_word[node] = selected_words_per_node
    nx.set_node_attributes(schemata_graph, schema_to_word,"schema_words")
    return schemata_graph

def decode_words_from_schema_walk(walk,schema_to_word):
    sentence = []
    for schema in walk:
        sentence.append(random.choice(schema_to_word[schema]))
    return sentence

def generate_simple_languages(schemata_graph,sentence_size_rate=15.3,number_of_sentences=100):
    sparse_adjacency = nx.adjacency_matrix(schemata_graph).tocoo()
    schema_to_word = nx.get_node_attributes(schemata_graph,'schema_words')
    sentences = []
    for ijk in range(number_of_sentences):
        sentence_size = poisson(sentence_size_rate).rvs()
        sentence_schemata_walk = node2vec_walk(sparse_adjacency,sentence_size)
        sentence = decode_words_from_schema_walk(sentence_schemata_walk,
                                                 schema_to_word)
        sentences.append(sentence)
    return sentences

def split_corpora(corpora,data_dir,schemata_name,test_size=.1,validation_size=.2):
    total_size = len(corpora)
    size_test = int(total_size*test_size)
    size_validation = int(total_size*validation_size)
    test_plus_validation = size_test+size_validation
    size_train = total_size - test_plus_validation
    training = corpora[:size_train]
    validation = corpora[size_train:size_train+size_validation]
    test =  corpora[size_train+size_validation:]

    f = open(os.path.join(data_dir,"{0}.train.txt".format(schemata_name)),"w")
    for sentence in training:
        f.write(" ".join(sentence)+"\n")
    f.close()
    f = open(os.path.join(data_dir,"{0}.test.txt".format(schemata_name)),"w")
    for sentence in test:
        f.write(" ".join(sentence)+"\n")
    f.close()
    f = open(os.path.join(data_dir,"{0}.valid.txt".format(schemata_name)),"w")
    for sentence in validation:
        f.write(" ".join(sentence)+"\n")
    f.close()
    return training, validation, test


if __name__=="__main__":
    schemata_name = "barabasi"
    data_dir = "C:/Users/cesar/Desktop/Projects/General/GENTEXT/sample_data/synthetic_languages/barabasi/"
    schemata_graph = generate_simple_schemata()
    nx.readwrite.write_gpickle(schemata_graph,os.path.join(data_dir,"{0}.schemata_graph.gp".format(schemata_name)))
    corpora = generate_simple_languages(schemata_graph, sentence_size_rate=15.3, number_of_sentences=1000)
    training, test, validation = split_corpora(corpora, data_dir,schemata_name, test_size=.1, validation_size=.2)

    generate_word2vec(corpora, corpora_name=schemata_name, embeddings_size=100, data_dir=data_dir)
    generate_word2vec(corpora, corpora_name=schemata_name, embeddings_size=50, data_dir=data_dir)
    generate_word2vec(corpora, corpora_name=schemata_name, embeddings_size=10, data_dir=data_dir)