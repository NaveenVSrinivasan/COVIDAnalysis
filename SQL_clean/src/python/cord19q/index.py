"""
Indexing module
"""

import os.path
import sqlite3
import sys

from .embeddings import Embeddings
from .models import Models
from .tokenizer import Tokenizer

class Index(object):
    """
    Methods to build a new sentence embeddings index.
    """

    @staticmethod
    def stream(dbfile):
        """
        Streams documents from an articles.sqlite file. This method is a generator and will yield a row at time.

        Args:
            dbfile: input SQLite file
        """

        # Connection to database file
        db = sqlite3.connect(dbfile)
        cur = db.cursor()

        # Select tagged sentences without a NLP label. NLP labels are set for non-informative sentences.
        cur.execute("SELECT Id, Text FROM sections WHERE tags is not null and labels is null")

        count = 0
        for row in cur:
            # Tokenize text
            tokens = Tokenizer.tokenize(row[1])

            document = (row[0], tokens, None)

            count += 1
            if count % 1000 == 0:
                print("Streamed %d documents" % (count))

            # Skip documents with no tokens parsed
            if tokens:
                yield document

        print("Iterated over %d total rows" % (count))

        # Free database resources
        db.close()

    @staticmethod
    def embeddings(dbfile, vectors):
        """
        Builds a sentence embeddings index.

        Args:
            dbfile: input SQLite file
            vectors: vector path

        Returns:
            embeddings index
        """

        embeddings = Embeddings({"path": vectors,
                                 "scoring": "bm25",
                                 "pca": 3})

        # Build scoring index if scoring method provided
        if embeddings.config["scoring"]:
            embeddings.score(Index.stream(dbfile))

        # Build embeddings index
        embeddings.index(Index.stream(dbfile))

        return embeddings

    @staticmethod
    def run(path, vectors):
        """
        Executes an index run.

        Args:
            path: model path, if None uses default path
            vectors: vector path, if None uses default path
        """

        # Default path if not provided
        if not path:
            path = Models.modelPath()

        dbfile = os.path.join(path, "articles.sqlite")

        # Default vectors
        if not vectors:
            vectors = Models.vectorPath("cord19-300d.magnitude")

        print("Building new model")
        embeddings = Index.embeddings(dbfile, vectors)
        embeddings.save(path)

if __name__ == "__main__":
    Index.run(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
