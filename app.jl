module Science

import Genie.Renderer.Json: json

using GenieFramework
using Genie.Renderer.Html
using Genie.Requests
using LibPQ, Jedis
using DataFrames, Tables
using SQLStrings
using JSON3

using PyCall
pushfirst!(pyimport("sys")."path", "modules")
transformer = pyimport("transformer")
redisearch = pyimport("redisearch")

include("modules/word2vec.jl")


config = JSON3.read(open("config.json", "r"))

memory = Jedis.Client(host = "localhost")

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
    ii = (; ii..., hostpass = Dict(pairs(NamedTuple{(:host,:pass)}(ii))))
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




function ask(question, model, context = "")
    pool = String(model.name) * ":" * replace(lowercase(question), " " => "-")
    check = Jedis.execute(["exists", pool], memory)
    if check == 0
        Jedis.execute(["zadd", pool, Base.time(), "START"], memory)
        formatted = "<|user|>\n" * question * " <|end|>\n<|assistant|>"    
        streamer = transformer.answer(model.tokenizer, model.transformer, formatted)
        i = 1
        for token in streamer
            if length(token) == 0
                token = "\n"
            end
            Jedis.execute(["zadd", pool, Base.time(), string(i) * "|" * token], memory)
            i = i + 1
        end
    end
end

function answer(question, model, start = 0)
    client = Jedis.Client(host = "localhost")
    Jedis.execute(["zrange", String(model.name) * ":" * replace(lowercase(question), " " => "-"), start, "-1"], client)
end


function metadata(results, fields = ["i"])
    global pqsql
    if "i" ∉ fields
        push!(fields, "i")
    end
    if length(results) > 0
        records = DataFrame([(i = parse(Int64, r.i), score = parse(Float32, r.score)) for r in results])
        result = LibPQ.execute(pqsql, "SELECT " * join(fields, ", ") * " FROM articles WHERE i IN (" * join([p.i for p in results], ", ") * ")")
        results = DataFrame(columntable(result))
        results = leftjoin(results, records, on = [:i])
        [NamedTuple(result) for result in eachrow(results)]
    else
        []
    end
end

function search(query, model; sources = [], dates = [], start = 0, N = 32)
    vector = model.name == :word2vec ? word2vec.sentence(query, model.redis) : model.encoder.encode(query)
    results = redisearch.search(vector, model.name, sources, dates, start, N, server = model.hostpass)
    metadata(results, ["id", "title", "year", "authors", "abstract", "source"])
end

function article(id)
    global pqsql
    meta = LibPQ.execute(pqsql, sql`SELECT i, id, title, year, authors, abstract, source FROM articles WHERE id = $id`)
    return rowtable(meta)[1]
end




function parseque(model, query)
    query = replace(query, "[answer]" => '?')
    sourci = String.(keys(sources))
    selection = Dict(sourci .=> ones(length(sources), 1))
    if occursin("[", model)
        model, over = split(model, "[")
        over = split(lowercase(chop(over)), ",")
        for source in sourci
            if source ∉ over
                selection[source] = 0
            end
        end
    end
    if occursin(model, ",")
        model, generator = split(model, ",")
    else
        generator = length(answering) > 0 ? first(keys(answering)) : ""
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
    [model, generator, query, selection, souse, seledates]
end



route("/", method = GET) do
   selection = Dict(String.(keys(sources)) .=> ones(length(sources), 1))
   html(path"app.jl.html", results = "", count = 0, query = "", N = n,
                           imodel = :word2vec, jmodel = length(answering) > 0 ? first(keys(answering)) : "",
                           retrieval = retrieval, generative = answering,
                           sources = selection, dates = dates, total = 11001479)
end


route("/:model/:query", method = GET) do
    model, generator, query, selection, souse, seledates = parseque(payload(:model), payload(:query))
    original = payload(:model) * "/" * payload(:query)
    if model in keys(retrieval)
        model = retrieval[model]
        elapsed = @elapsed results = search(query, model, sources = souse, dates = seledates)
        html(path"app.jl.html", query = query,         results = results,     count = length(results),    time = elapsed,
                                imodel = model.name,   retrieval = retrieval, jmodel = generator, generative = answering,
                                sources = selection, dates = seledates, request = original)
    elseif  model in keys(answering)
        # 
    end
end

route("/scroll/:model/:query/:start::Int", method = GET) do
    model, generator, query, selection, souse, seledates = parseque(payload(:model), payload(:query))
    if model in keys(retrieval)
        search(query, retrieval[model], sources = souse, dates = seledates, start = payload(:start)) |> json
    end
end

route("/ask/:generator/:question", method = GET) do
    model = answering[Symbol(payload(:generator))]
    ask(payload(:question) * "?", model)
end

route("/answer/:generator/:question/:start::Int", method = GET) do
    model = answering[Symbol(payload(:generator))]
    answer(payload(:question) * "?", model, payload(:start)) |> json
end


route("/article/:id", method = GET) do
    ai = article(payload(:id))
    ai = (; ai..., url    = sources[ai.source].url * ai.id,
                   source = sources[ai.source].name)

    model = retrieval[:gtelarge]
    similar = redisearch.similar(ai.i, model.name, server = model.hostpass)
    similar = metadata(similar, ["id", "title", "year"])
    similar = similar[2:length(similar)]
    
    html(path"art.jl.html", article = ai, similar = similar)
end


route("/favicon.ico") do 
    redirect("/img/re.ico")
end

Server.up(1234)

end
