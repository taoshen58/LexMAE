import os
import sys
import collections
import csv
import copy
import logging
import re
import regex
import unicodedata
from peach.common import load_tsv, save_list_to_file, load_json, save_json, get_line_offsets, file_exists
import datasets

csv.field_size_limit(sys.maxsize)


def _normalize(text):
    return unicodedata.normalize('NFD', text)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None

def load_trec(trec_file, int_id=True):
    trans_fn = int if int_id else lambda a: a

    logging.info(f"loading trec {trec_file}")
    qid2results = dict()  # {qid: {pid: {score: , rank: }}}
    with open(trec_file) as fp:
        for idx_line, line in enumerate(fp):
            try:
                qid, _, pid, rank, score, _method = line.strip().split(" ")
            except ValueError:
                print(f"Please check line {idx_line} in {trec_file}")
                raise ValueError
            qid, pid, rank, score = trans_fn(qid), trans_fn(pid), int(rank), float(score)
            # if rank <= TOP_K:  # only count TOP K  TAO: add support
            if qid not in qid2results:  # initialize
                qid2results[qid] = collections.OrderedDict()
            qid2results[qid][pid] = score
    return qid2results


def save_qid2results_to_trec(
        qid2results, top_k=1000, source_name="NoSourceName", save_to_file=None, ):
    trec_str_list = []
    all_qid_list = list(sorted(qid2results.keys()))
    for qid in all_qid_list:
        results = [(pid, score) for pid, score in qid2results[qid].items()]
        results.sort(key=lambda e: e[1], reverse=True)
        cur_results = [f"{qid} Q0 {pid} {idx+1} {score} {source_name}" for idx, (pid, score) in enumerate(results)]
        if len(cur_results) < top_k:
            logging.info(f"WARN: qid-{qid} only has {len(cur_results)} passage results, less than {top_k}!")
        trec_str_list.extend(cur_results[:top_k])
    logging.info(f"Trec from {source_name}, num of line is {len(trec_str_list)}")
    if save_to_file is not None:
        logging.info(f"save to {save_to_file}")
        save_list_to_file(trec_str_list, save_to_file)
    return trec_str_list

def transform_qid2results_to_qid2hn(qid2results, top_k, qrels_path, int_id=True):
    trans_fn = int if int_id else lambda a: a

    qrels = load_tsv(os.path.join(qrels_path))
    qid2pos_pids = collections.defaultdict(set)
    for qrel in qrels:
        assert len(qrel) == 4
        qid, pid = trans_fn(qrel[0]), trans_fn(qrel[2])  # todo: judge if > 0
        qid2pos_pids[qid].add(pid)

    qid2negatives = dict()
    all_qid_list = list(sorted(qid2results.keys()))
    for qid in all_qid_list:
        qid = trans_fn(qid)
        pos_pids = qid2pos_pids[qid]
        results = [(pid, score) for pid, score in qid2results[qid].items()]
        results.sort(key=lambda e: e[1], reverse=True)
        negs = [trans_fn(pid) for pid, s in results if trans_fn(pid) not in pos_pids]
        if len(negs) < top_k:
            logging.info(f"WARN: qid-{qid} only has {len(negs)} passage results, less than {top_k}!")
        qid2negatives[qid] = negs[:top_k]
    return qid2negatives

def calculate_metrics_for_qid2results(qid2results, qrels_path, int_id=True):
    trans_fn = int if int_id else lambda a: a
    import datasets
    from tqdm import tqdm
    metric_ranking = datasets.load_metric("peach/metrics/ranking_v3.py")
    metric_ranking.qids_list = []
    for qid, results in tqdm(qid2results.items(), desc="Calculating metrics ..."):
        pids, scores = zip(*[(trans_fn(pid), score) for pid, score in results.items()])
        metric_ranking.add_batch(predictions=scores,
                                 references=[str(pid) for pid in pids], # v3 compatibility
                                 )
        metric_ranking.qids_list.extend([trans_fn(qid)] * len(results))
    eval_metrics = metric_ranking.compute(
        group_labels=metric_ranking.qids_list,
        qrels_path=qrels_path,
    )
    return eval_metrics


def calculate_metrics_for_qid2results_with_qrefs(qid2results, qrefs_path, int_id=True):
    trans_fn = int if int_id else lambda a: a

    qid2refs = load_qrefs(qrefs_path)

    simple_tokenizer = SimpleTokenizer()

    # load collection.tsv.offset.json
    collection_path = os.path.join(os.path.dirname(qrefs_path), "../collection.tsv")
    pid2offset_file_path = collection_path + ".pid2offset.json"
    if file_exists(pid2offset_file_path):
        pid2offset = dict((trans_fn(k), v) for k, v in load_json(pid2offset_file_path).items())
    else:
        offset_file_path = collection_path + ".offset.json"
        if file_exists(offset_file_path):
            offsets = load_json(offset_file_path)
        else:
            offsets = get_line_offsets(collection_path, encoding="utf-8")
            save_json(offsets, offset_file_path)
        pid2offset = dict()
        with open(collection_path, encoding="utf-8") as fp:
            for idx_line, line in enumerate(fp):
                passage_id, _ = line.strip("\n").split("\t")
                pid2offset[trans_fn(passage_id)] = offsets[idx_line]
            assert len(pid2offset) == len(offsets)
            save_json(pid2offset, pid2offset_file_path)
    collection_pid2offset = pid2offset

    all_qids = list(qid2results.keys())
    all_qids.sort()

    metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
    metric_ranking.qids_list = []

    collection_fp = open(collection_path, encoding="utf-8")
    for qid in all_qids:
        refs = qid2refs[qid]  # assert qid in qid2refs
        pid2score = qid2results[qid]
        top_scores, top_references = [], []
        for pid, score in pid2score.items():
            # get label
            label = 0
            collection_fp.seek(collection_pid2offset[pid])
            line = collection_fp.readline()
            passage_id, passage = line.strip("\n").split("\t")
            assert trans_fn(passage_id) == pid
            passage_words = simple_tokenizer.tokenize(_normalize(passage)).words(uncased=True)
            for single_ref in refs:
                single_ref_words = simple_tokenizer.tokenize(_normalize(single_ref)).words(uncased=True)
                for i in range(0, len(passage_words) - len(single_ref_words) + 1):
                    if single_ref_words == passage_words[i: i + len(single_ref_words)]:
                        label = 1
                        break
                if label == 1:
                    break
            top_scores.append(score)
            top_references.append(label)
            # add to metric script (pid, qid, score, label)
        metric_ranking.add_batch(predictions=top_scores, references=top_references)
        metric_ranking.qids_list.extend([qid] * len(pid2score))
    collection_fp.close()
    eval_metrics = metric_ranking.compute(group_labels=metric_ranking.qids_list)
    return eval_metrics

def load_qrefs(qaref_file_path, int_id=True):
    trans_fn = int if int_id else lambda a: a

    qid2refs = dict()
    with open(qaref_file_path) as fp:
        for line in fp:
            data = line.strip("\n").split("\t")
            assert len(data) >= 2
            qid2refs[trans_fn(data[0])] = data[1:]
    return qid2refs


def evaluate_trec_file(trec_file_path, qrels_file_path, int_id=True):
    qid2results = load_trec(trec_file_path, int_id=int_id)
    if os.path.exists(qrels_file_path) and os.path.isfile(qrels_file_path):
        eval_metrics = calculate_metrics_for_qid2results(qid2results, qrels_file_path, int_id=int_id)
    else:
        qrefs_file_path = f"{qrels_file_path}.qrefs.tsv"  # qid \t ref1 \t ref2 ... \n
        eval_metrics = calculate_metrics_for_qid2results_with_qrefs(qid2results, qrefs_file_path)

    return eval_metrics


# ============== simple tokenizer ======================




class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logging.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)