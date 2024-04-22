module Science

using GenieFramework
using Genie.Renderer.Html, Genie.Requests
using WordTokenizers, Statistics
using LibPQ, Tables, Jedis
using PyCall

redis = Client(host = "localhost", port = 6379)
datac = LibPQ.Connection("dbname=science host=localhost user=researchers password=KRASLApQ6QjE6hX6ff")

println("connected to databases.")

#module Pgvector
#    convert(v::AbstractVector{T}) where T<:Real = string("[", join(v, ","), "]")
#end

function wordvector(word, N)
    global redis

    vector = Jedis.hget("embeddings", word; client = redis)
    if isnothing(vector)
        vector = Jedis.hget("embeddings", lowercase(word); client = redis)
    end
    if isnothing(vector)
        Float32.(vec(zeros(1, N)))
    else
        parse.(Float32,split(chop(vector,head=1),","))
    end
end

function getvector(title, text)
    words = reduce(vcat, nltk_word_tokenize.(split_sentences(title * "\n" * text)))
    vectors = stack(wordvector.(words, 300))
    mean(vectors, dims = 2)
end


py"""

import redis
import numpy
from redis.commands.search.query import Query

def runquery(vector):
    query = (
        Query("(*)=>[KNN 10000 @embedding $vector AS score]")
        .sort_by("score")
        .paging(0, 32)
        .return_fields("score", "i")
        .dialect(2)
    )
    bytes = numpy.array(vector, dtype = numpy.float32).tobytes()
    client = redis.Redis(port = 6379, decode_responses = True)
    result = client.ft("word2vec").search(query, {"vector": bytes})
    return result.docs

"""



route("/") do
   html(path"app.jl.html", results = "", query = "")
end

route("/", method = POST) do
   query = postpayload(:query)
   qvector = vec(getvector("", query))
#   embedding = Pgvector.convert(qvector)
   results = py"runquery"(qvector)
   
   ids = join([p.i for p in results], ", ")
   scores = Dict(p.i => p.score for p in results)
   scores = sort(collect(scores), by = i -> i[2])

   # result = LibPQ.execute(datac, "SELECT id, title, year, abstract FROM arxiv ORDER BY embedding <=> \$1 LIMIT 32", [embedding])
   result = LibPQ.execute(datac, "SELECT i, id, title, year, abstract FROM arxiv WHERE i IN (" * ids * ")")
   results = rowtable(result)

   metadata = [p.i => (p.id, p.title, p.year, p.abstract, scores[string(p.i)]) for p in results]
   metadata = values(Dict(sort(collect(metadata), by = i -> i[2][4])))

   html(path"app.jl.html", query = query, results = metadata)
end


Server.up(8000)

end
