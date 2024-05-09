import redis, numpy
from redis import Redis
from redis.commands.search.query import Query

def reconn(server):
    return Redis(host = server['host'], password = server['pass'], decode_responses = True) if server else Redis(decode_responses = True)

def search(vector, model, sources = [], dates = [], start = 0, N = 32, server = None):
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
    result = reconn(server).ft(model).search(query, {"vector": bytes})
    return result.docs

def similar(i, model, N = 12, server = None):
    query = (Query("@i:[" + str(i) + " " + str(i) + "]").return_fields(model))
    vector = eval(reconn(server).ft(model).search(query).docs[0][model])
    return search(vector, model, N = N, server = server)
