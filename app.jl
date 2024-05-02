module Science

using GenieFramework
using Genie.Renderer.Html
using Genie.Requests
using WordTokenizers, Statistics
using LibPQ, Tables, Jedis
using DataFrames
using PyCall

import Genie.Renderer.Json: json

redis = Client(host = "127.0.0.1", port = 6379)
pqsql = LibPQ.Connection("dbname=science host=localhost user=researchers password=KRASLApQ6QjE6hX6ff")

println("connected to databases.")


py"""
def sentenceTrans(path):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, trust_remote_code = True, device = "cuda")

def generator(path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to("cuda")
    return tokenizer, model
"""

gtemodel = py"sentenceTrans"("Alibaba-NLP/gte-large-en-v1.5")
expmodel = py"generator"("yam-peleg/Experiment21-7B")

println("language models loaded.")


function gtelarge(query)

    global gtemodel
    return gtemodel.encode(query)

end

function word2vec(query)

    function wordvec(word, N)
        global redis
        vector = Jedis.hget("vectors:word2vec", word; client = redis)
        if isnothing(vector)
            vector = Jedis.hget("vectors:word2vec", lowercase(word); client = redis)
        end
        if isnothing(vector)
            Float32.(vec(zeros(1, N)))
        else
            parse.(Float32,split(chop(vector,head=1),","))
        end
    end

    words = reduce(vcat, nltk_word_tokenize.(split_sentences(query)))
    vectors = stack(wordvec.(words, 300))
    vec(mean(vectors, dims = 2))

end


py"""

import redis
import numpy
from redis.commands.search.query import Query

def runquery(vector, model, start, N):
    query = (
        Query("(*)=>[KNN 8192 @" + model + " $vector AS score]")
        .sort_by("score")
        .paging(start, N)
        .return_fields("score", "i")
        .dialect(2)
    )
    bytes = numpy.array(vector, dtype = numpy.float32).tobytes()
    client = redis.Redis(port = 6379, decode_responses = True)
    result = client.ft(model).search(query, {"vector": bytes})
    return result.docs

def answer(tokenizer, model, question):
    inputs = tokenizer(question, return_tensors = "pt", return_attention_mask = False).to("cuda")
    outputs = model.generate(**inputs, max_length = 1024)
    text = tokenizer.batch_decode(outputs)[0]
    return text

"""

function answer(question)
    global expmodel
    tokenizer, model = expmodel
    py"answer"(tokenizer, model, question)
end

function search(query, model, start = 0, N = 32)
    global pqsql

    models = Dict("word2vec"          => :word2vec,
                  "gte-large-en-v1.5" => :gtelarge)

    if model in keys(models)
        model = models[model]
        vector = getfield(Science, model)(query)
        results = py"runquery"(vector, model, start, N)
        records = DataFrame([NamedTuple([:i => parse(Int, r.i), :score => parse(Float32, r.score)])  for r in results])
        result = LibPQ.execute(pqsql, "SELECT i, id, title, year, abstract FROM arxiv WHERE i IN (" * join([p.i for p in results], ", ") * ")")
        results = DataFrame(columntable(result))
        results = leftjoin(results, records, on = [:i])
        [NamedTuple(result) for result in eachrow(results)]
    
    end
end


route("/", method = GET) do
   html(path"app.jl.html", results = "", query = "", count = 0)
end

route("/:model/:query", method = GET) do
   model = payload(:model)
   query = payload(:query)
   elapsed = @elapsed results = search(query, model)
   html(path"app.jl.html", query = query, results = results, model = model, count = length(results), time = elapsed)
end

route("/scroll/:model/:query/:start", method = GET) do
    model = payload(:model)
    query = payload(:query)
    start = payload(:start)
    search(query, model, start) |> json
end

route("/answer/:question", method = GET) do
    question = payload(:question)
    answer(question) |> json
end

route("/favicon.ico") do 
    redirect("/img/re.ico")
end

Server.up(1234)

end
