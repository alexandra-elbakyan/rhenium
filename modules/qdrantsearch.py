from qdrant_client import QdrantClient, models


def search(vector, model, sources = [], dates = [], start = 0, N = 32, server = None):

    must = []

    if len(sources):
        must.append(models.FieldCondition(
                key = "source",
                match = models.MatchAny(
                    any = sources,
                ),
            ))
        
    if len(dates):
        must.append(models.FieldCondition(
                key = "year",
                range = models.Range(
                    gte = dates[0],
                    lte = dates[1],
                )
            ))

    filter = models.Filter(must = must) if len(must) else None

    client = QdrantClient(url = "https://" + server['host'], api_key = server['pass'])

    results = client.search(
        collection_name = model,
        query_vector = vector,
        query_filter = filter,
        with_vectors = False,
        with_payload = False,
        limit = N,
        offset = start,
    )

    class match:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    matches = []
    for result in results:
        ri = {'i' : result.id, 'score' : 1 - result.score}
        matches.append(match(**ri))

    return matches
