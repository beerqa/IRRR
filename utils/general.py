
def chunks(iterator, n):
    """Yield successive n-sized chunks from iterator."""
    chunk = []
    for x in iterator:
        chunk.append(x)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

def make_context(question, ir_results):
    """
    Creates a single context string.

    :question: string
    :ir_results: list of dictionaries objects each of which
        should have 'title' and 'text'
        (e.g. each entry of result from bulk_text_query)
    """
    return question + ' ' + concat_paragraphs(ir_results)

def concat_paragraphs(ir_results):
    return ' '.join([f"<t> {p['title']} </t> {''.join(p['text'])}" for p in ir_results])
