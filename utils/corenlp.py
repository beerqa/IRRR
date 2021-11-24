from stanza.server import CoreNLPClient, StartServer
from stanza.server.client import AnnotationException

LIMIT = 10000000
tokenizer_client = CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=300000, memory='16G', properties={'tokenize.ptb3Escaping': False, 'ssplit.eolonly': True, 'tokenize.options': "splitHyphenated=true"}, server_id='pengqi', max_char_length=LIMIT, threads=16, start_server=StartServer.TRY_START, be_quiet=True)

def bulk_tokenize(text, return_offsets=False):
    try:
        return _bulk_tokenize(text, return_offsets=return_offsets)
    except AnnotationException:
        # text is too long
        tokenized = []
        offsets = []

        chunk = []
        chunksize = 0
        for line in text:
            if chunksize + len(line) > LIMIT:
                res = _bulk_tokenize(chunk, return_offsets=return_offsets)
                if return_offsets:
                    tokenized.extend(res[0])
                    offsets.extend(res[1])
                else:
                    tokenized.extend(res)
                chunk = []
                chunksize = 0

            chunk.append(line)
            chunksize += len(line) + 1

        if chunksize > 0:
            res = _bulk_tokenize(chunk, return_offsets=return_offsets)
            if return_offsets:
                tokenized.extend(res[0])
                offsets.extend(res[1])
            else:
                tokenized.extend(res)

        if return_offsets:
            return tokenized, offsets
        else:
            return tokenized

def _bulk_tokenize(text, return_offsets=False):
    ann = tokenizer_client.annotate('\n'.join(text))

    if return_offsets:
        return [[token.originalText for token in sentence.token] for sentence in ann.sentence], [[(token.beginChar, token.endChar) for token in sentence.token] for sentence in ann.sentence]
    else:
        return [[token.originalText for token in sentence.token] for sentence in ann.sentence]

def get_sentence_list(text):
    # simple heuristic based sentence segmentation
    sentences = []
    splitted = text.split('. ')
    for i, s in enumerate(splitted):
        current = s if i == len(splitted) - 1 else (s + '. ')
        if len(sentences) > 0 and len(sentences[-1]) > 3 and sentences[-1][-3].isupper():
            sentences[-1] += current
        else:
            sentences.append(current)
    return sentences

if __name__ == "__main__":
    print(bulk_tokenize(['This is a test sentence.', ' This is another.']))
