import unittest
from trie import Trie

class TrieTest(unittest.TestCase):

    def setUp(self) -> None:
        self.trie = Trie()
        self.words = ["abc", "def", "abcd", "ghijklm"]
        for word in self.words:
            self.trie.insert(word)

    def test_exists(self):
        self.assertTrue(self.trie.search('abc'))
        self.assertTrue(self.trie.search('abcd'))
        self.assertTrue(self.trie.search('def'))
        self.assertFalse(self.trie.search(''))
        self.assertFalse(self.trie.search('abbd'))
        self.assertFalse(self.trie.search('ghijkl'))

    def test_prefix_exists(self):
        self.assertTrue(self.trie.prefix_search('ab'))
        self.assertTrue(self.trie.prefix_search('ghi'))
        self.assertFalse(self.trie.prefix_search(''))
        self.assertTrue(self.trie.prefix_search('abcd'))
        self.assertFalse(self.trie.prefix_search('abcdko'))

    def test_all_suffixes(self):
        self.assertListEqual(list(self.trie.all_suffixes(self.trie.root)), sorted(self.words))
        # TODO: add some more tests

    def test_autocomplete(self):
        f = lambda x: sorted(list(self.trie.autocomplete(x)))
        self.assertListEqual(f('ab'), sorted(["abc", "abcd"]))
        self.assertListEqual(f('d'), sorted(["def"]))
        self.assertListEqual(f('abcd'), sorted(["abcd"]))
        self.assertListEqual(f(''), sorted(self.words))

    def test_delete(self):
        self.trie.insert("apple")
        self.trie.delete("apple")
        self.assertRaises(ValueError, self.trie.delete, "apple")
        self.assertFalse(self.trie.search("apple"))

