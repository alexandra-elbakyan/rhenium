module Science

import Genie.Renderer.Json: json

using GenieFramework
using Genie.Renderer.Html
using Genie.Requests
using LibPQ, Jedis
using DataFrames, Tables
using JSON3

using PyCall
pushfirst!(pyimport("sys")."path", "modules")
transformer = pyimport("transformer")
redisearch = pyimport("redisearch")

include("modules/word2vec.jl")


config = JSON3.read(open("config.json", "r"))


print("connecting postgresql database... ")
pqsql = LibPQ.Connection("dbname=$(config.postgres.database) host=$(config.postgres.host) user=$(config.postgres.user) password=$(config.postgres.pass)")
println("done.")

sources = config.sources

dates = [1974, 2024]
n = 0

println("loading retrieval models:")
retrieval = Dict()
for mi in keys(config.retrieval)
    ii = (; config.retrieval[mi]..., name = mi)
    if mi == :word2vec
        ii = (; ii..., redis = Jedis.Client(host = ii.host))
        if !isempty(ii.pass)
            Jedis.auth(ii.pass, client = ii.redis)
            n = Jedis.execute(["ft.info", "word2vec"], ii.redis)[10]
        end
        ii = (; ii..., id = "word2vec")
    else
        ii = (; ii..., encoder = transformer.sentence(ii.path))
        ii = (; ii..., id = basename(ii.path))
    end
    retrieval[mi] = ii
    println("[+] " * String(mi))
end
println("done!")

println("articles indexed: " * string(n))

println("loading generative models:")
answering = Dict()
for mi in keys(config.answering)
    ii = (; config.answering[mi]..., name = mi)
    tk, tf = transformer.generator(ii.path)
    ii = (; ii..., tokenizer = tk, transformer = tf)
    ii = (; ii..., id = basename(ii.path))
    answering[mi] = ii
    println("[+] " * String(mi))
end
println("done!")





function answer(question, model)
    response = transformer.answer(model.tokenizer, model.transformer, question)
    response = chop(response, head = 4 + length(question) + 1, tail = 4)
    if occursin("INST", response)
        response = split(response, "INST")[1]
    end
    response = split(response, ".")
    join(response[1:length(response)-1], ".") * "..."
end


function search(query, model; sources = [], dates = [], start = 0, N = 32)
    global pqsql
    vector = model.name == :word2vec ? word2vec.sentence(query, model.redis) : model.encoder.encode(query)
    results = redisearch.runquery(vector, model.name, sources, dates, start, N, server = Dict(pairs(NamedTuple{(:host,:pass)}(model))))
    if length(results) > 0
        records = DataFrame([NamedTuple([:i => parse(Int, r.i), :score => parse(Float32, r.score)])  for r in results])
        result = LibPQ.execute(pqsql, "SELECT i, id, title, year, authors, abstract, source FROM articles WHERE i IN (" * join([p.i for p in results], ", ") * ")")
        results = DataFrame(columntable(result))
        results = leftjoin(results, records, on = [:i])
        [NamedTuple(result) for result in eachrow(results)]
    else
        []
    end
end


function parseque(model, query)
    query = replace(query, "[answer]" => '?')
    selection = Dict(sources .=> ones(length(sources), 1))
    if occursin("[", model)
        model, over = split(model, "[")
        over = split(lowercase(chop(over)), ",")
        for source in sources
            if source ∉ over
                selection[source] = 0
            end
        end
    end
    seledates = dates
    if occursin("[", query)
        query, years = split(query, "[")
        years = tryparse.(Int64, split(chop(years), "-"))
        if (years[1] > 1000) & (years[2] < 9999)
            seledates = years
        end
    end
    souse = collect(keys(filter( ab -> ab[2] > 0, selection)))
    if length(souse) == length(sources)
        souse = []
    end
    query = strip(query)
    model = Symbol(model)
    [model, query, selection, souse, seledates]
end



route("/", method = GET) do
   selection = Dict(sources .=> ones(length(sources), 1))
   html(path"app.jl.html", results = "", query = "", N = n,
                           imodel = :word2vec, retrieval = retrieval,
                           sources = selection, dates = dates, count = 11001479)
end


route("/:model/:query", method = GET) do
    model, query, selection, souse, seledates = parseque(payload(:model), payload(:query))
    original = payload(:model) * "/" * payload(:query)
    if model in keys(retrieval)
        model = retrieval[model]
        elapsed = @elapsed results = search(query, model, sources = souse, dates = seledates)
        html(path"app.jl.html", query = query,         results = results,     count = length(results),    time = elapsed,
                                imodel = model.name,   retrieval = retrieval, jmodel = "experiment21-7B", generative = answering,
                                sources = selection, dates = seledates, request = original)
    end
end

route("/scroll/:model/:query/:start::Int", method = GET) do
    model, query, selection, souse, seledates = parseque(payload(:model), payload(:query))
    if model in keys(retrieval)
        search(query, retrieval[model], sources = souse, dates = seledates, start = payload(:start)) |> json
    end
end

route("/answer/:question", method = GET) do
    question = payload(:question)
    question = replace(question, "[answer]" => '?')
    answer(question, answering[Symbol("experiment21-7B")])
end

route("/favicon.ico") do 
    redirect("/img/re.ico")
end

Server.up(1234)

end
