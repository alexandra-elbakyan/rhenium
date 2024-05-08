import redis, numpy
from redis.commands.search.query import Query

def runquery(vector, model, sources = [], dates = [], start = 0, N = 32, server = None):
    params = []
    if len(sources): params.append("@source:{" + "|".join(sources) + "}")
    if len(dates): params.append("@year:[" + " ".join([str(y) for y in dates]) + "]")
    params =  ' '.join(params) if params else '*'
    query = (
        Query("(" + params + ")=>[KNN 512 @" + model + " $vector AS score]")
        .sort_by("score")
        .paging(start, N)
        .return_fields("score", "i")
        .dialect(2)
    )
    bytes = numpy.array(vector, dtype = numpy.float32).tobytes()
    if server:
        client = redis.Redis(host = server['host'], password = server['pass'], decode_responses = True)
    else:
        client = redis.Redis(decode_responses = True)
    result = client.ft(model).search(query, {"vector": bytes})
    return result.docs
