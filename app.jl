module Science

using GenieFramework
using Genie.Renderer.Html, Genie.Requests
using WordTokenizers, Statistics
using LibPQ, Tables, Jedis

redis = Client(host = "localhost", port = 6379)
datac = LibPQ.Connection("dbname=science host=localhost user=researchers password=KRASLApQ6QjE6hX6ff")

println("connected to databases.")

module Pgvector
    convert(v::AbstractVector{T}) where T<:Real = string("[", join(v, ","), "]")
end

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


route("/") do
   html(path"app.jl.html", results = "", query = "")
end

route("/", method = POST) do
   query = postpayload(:query)
   qvector = vec(getvector("", query))
   embedding = Pgvector.convert(qvector)
   result = LibPQ.execute(datac, "SELECT id, title, year, abstract FROM arxiv ORDER BY embedding <=> \$1 LIMIT 32", [embedding])
   results = rowtable(result)
   html(path"app.jl.html", query = query, results = results)
end


Server.up(8000)

end
