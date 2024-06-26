module Science

import Genie.Renderer.Json: json

using GenieFramework
using Genie.Renderer.Html
using Genie.Requests
using LibPQ, Jedis
using DataFrames, Tables
using SQLStrings
using Format
using JSON3
using CSV

using PyCall
pushfirst!(pyimport("sys")."path", "modules")
transformer = pyimport("transformer")
qdrantsearch = pyimport("qdrantsearch")
redisearch = pyimport("redisearch")

include("modules/word2vec.jl")


config = JSON3.read(open("config.json", "r"))

memory = Jedis.Client(host = "localhost")

print("connecting postgresql database... ")
pqsql = LibPQ.Connection("dbname=$(config.postgres.database) host=$(config.postgres.host) user=$(config.postgres.user) password=$(config.postgres.pass)")
println("done.")

sources = config.sources

println("loading retrieval models:")
retrieval = Dict()
for mi in keys(config.retrieval)
    ii = (; config.retrieval[mi]..., name = mi)
    if (haskey(ii, :enabled) && !ii.enabled)
        continue
    end
    ii = (; ii..., hostpass = Dict(pairs(NamedTuple{(:host,:pass)}(ii))))
    if mi == :word2vec
        ii = (; ii..., redis = Jedis.Client(host = ii.host))
        if haskey(ii, :pass)
            Jedis.auth(ii.pass, client = ii.redis)
        end
        ii = (; ii..., n = Jedis.execute(["dbsize"], ii.redis))
        ii = (; ii..., id = "word2vec")
    else
        ii = (; ii..., encoder = transformer.sentence(ii.path))
        if (!haskey(ii, :id))
            ii = (; ii..., id = basename(ii.path))
        end
    end
    retrieval[mi] = ii
    println("[+] " * String(mi))
end
println("done!")

timespan = [1963, 2024]

n = retrieval[:word2vec].n
println("articles indexed: " * string(n))

println("loading generative models:")
answering = Dict()
for mi in keys(config.answering)
    ii = (; config.answering[mi]..., name = mi)
    if (haskey(ii, :enabled) && !ii.enabled)
        continue
    end
    quantize = haskey(ii, :quantize) ? ii.quantize : nothing
    tk, tf = transformer.generator(ii.path, quanti = quantize)
    ii = (; ii..., tokenizer = tk, transformer = tf)
    ii = (; ii..., id = basename(ii.path))
    answering[mi] = ii
    println("[+] " * String(mi))
end
println("done!")





function ask(question, model; context = "", prefix = ":")
    pool = String(model.name) * prefix * replace(lowercase(question), " " => "-")
    check = Jedis.execute(["exists", pool], memory)
    if check == 0
        Jedis.execute(["zadd", pool, Base.time(), "<|startup|>"], memory)
        if haskey(model, :format)
            formatted = replace(model.format, "{question}" => question)
            formatted = replace(formatted, "{context}" => context)
        else
            formatted = question
            if length(context) > 0
                formatted = "Please answer the question using the following contextual information:\n\n" * context * "\n\nQuestion:" * question
            end
        end   
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

function answer(question, model; start = 0, prefix = ":")
    client = Jedis.Client(host = "localhost")
    Jedis.execute(["zrange", String(model.name) * prefix * replace(lowercase(question), " " => "-"), start, "-1"], client)
end


function metadata(results, fields = ["i"])
    global pqsql
    if "i" ∉ fields
        push!(fields, "i")
    end
    if length(results) > 0
        records = []
        for r in results
            i = typeof(r.i) == String ? parse(Int64, r.i) : r.i
            score = typeof(r.score) == String ? parse(Float32, r.score) : r.score
            push!(records, (i = i, score = score))
        end
        records = DataFrame(records)
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
    searchf = haskey(model, :field) ? qdrantsearch.search : redisearch.search
    results = searchf(vector, lowercase(String(model.name)), sources, dates, start, N, server = model.hostpass)
    metadata(results, ["id", "title", "year", "authors", "abstract", "source"])
end

function article(id)
    global pqsql
    meta = LibPQ.execute(pqsql, sql`SELECT i, id, title, year, authors, abstract, source FROM articles WHERE id = $id`)
    return rowtable(meta)[1]
end




function parseque(model, query)
    RAG = true
    israg = Dict("RAG" => true, "answer" => false)
    for ra in keys(israg)
        if occursin("[" * ra * "]", query)
            query = replace(replace(query, "[" * ra * "]" => '?'), " ?" => '?')
            RAG = israg[ra]
        end
    end
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
    if occursin(",", model)
        model, generator = split(model, ",")
    else
        generator = length(answering) > 0 ? first(keys(answering)) : ""
    end
    dates = []
    if occursin("[", query)
        query, years = split(query, "[")
        years = tryparse.(Int64, split(chop(years), "-"))
        if (years[1] > 1000) & (years[2] < 9999)
            dates = years
        end
    end
    databases = collect(keys(filter( ab -> ab[2] > 0, selection)))
    if length(databases) == length(sources)
        databases = []
    end
    query = strip(query)
    model = Symbol(model)
    generator = Symbol(generator)
    [model, generator, query, selection, databases, dates, RAG]
end



route("/", method = GET) do
   examples = CSV.File("searches.csv") |> DataFrame
   categories = Array(unique(examples[!, [:category]]))
   groups = Dict()
   for category in categories
       groups[category] = NamedTuple.(eachrow(examples[in([category]).(examples.category), :]))
   end
   imodel = length(retrieval) > 0 ? first(keys(retrieval)) : ""
   jmodel = length(answering) > 0 ? first(keys(answering)) : ""
   selection = Dict(String.(keys(sources)) .=> ones(length(sources), 1))
   html(path"app.jl.html", results = "", count = 0, query = "", N = n,
                           imodel = imodel, jmodel = jmodel, examples = groups,
                           retrieval = retrieval, generative = answering, RAG = true,
                           sources = selection, dates = timespan, total = cfmt("%\'d", n))
end


route("/:model/:query", method = GET) do
    model, generator, query, selection, databases, dates, RAG = parseque(payload(:model), payload(:query))
    original = payload(:model) * "/" * payload(:query)
    if model in keys(retrieval)
        model = retrieval[model]
        elapsed = @elapsed results = search(query, model, sources = databases, dates = dates)
        html(path"app.jl.html", query = query,         results = results,     count = length(results),    time = elapsed,
                                imodel = model.name,   retrieval = retrieval, jmodel = generator, generative = answering,
                                sources = selection, dates = isempty(dates) ? timespan : dates, request = original, RAG = RAG)
    end
end

route("/scroll/:model/:query/:start::Int", method = GET) do
    model, generator, query, selection, databases, dates, RAG = parseque(payload(:model), payload(:query))
    if model in keys(retrieval)
        search(query, retrieval[model], sources = databases, dates = dates, start = payload(:start)) |> json
    end
end

route("/ask/:model/:question", method = GET) do
    retriever, generator, question, _, databases, dates, RAG = parseque(payload(:model), payload(:question))
    if RAG
        topN = search(question, retrieval[retriever], sources = databases, dates = dates, start = 0, N = 8)
        context = ""
        for ai in topN
            context = context * ai.title * "\n" * ai.abstract * "\n\n"
        end
        prefix = ":" * String(retriever) * ":" * join(databases, "|") * ":" * join(dates, "-") * ":"
    else
        context = ""
        prefix = ":"
    end
    ask(question, answering[Symbol(generator)], context = context, prefix = prefix)
end

route("/answer/:model/:question/:start::Int", method = GET) do
    retriever, generator, question, _, databases, dates, RAG = parseque(payload(:model), payload(:question))
    prefix = RAG ? ":" * String(retriever) * ":" * join(databases, "|") * ":" * join(dates, "-") * ":" : ":"
    answer(question, answering[Symbol(generator)], start = payload(:start), prefix = prefix) |> json
end


route("/article/:id", method = GET) do
    ai = article(replace(payload(:id), "--" => "/"))
    ai = (; ai..., url    = sources[ai.source].url * ai.id,
                   source = sources[ai.source].name)

    for model in values(retrieval)
        if !haskey(model, :field)
            similar = redisearch.similar(ai.i, model.name, server = model.hostpass)
            similar = metadata(similar, ["id", "title", "year"])
            similar = similar[2:length(similar)]
            
            return html(path"art.jl.html", article = ai, similar = similar)
        end
    end
    

end


route("/favicon.ico") do 
    redirect("/img/re.ico")
end

Server.up(1234)

end