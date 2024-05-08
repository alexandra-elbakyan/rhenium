module word2vec

    using Jedis
    using WordTokenizers, Statistics

    function word(w::String, dictionary, N = 300)
        v = isnothing(dictionary[w]) ?
            dictionary[lowercase(w)] :
            dictionary[w]
        if v isa String
            parse.(Float32, split(chop(v, head = 1), ","))
        else
            Float32.(vec(zeros(1, N)))
        end
    end

    function sentence(s::AbstractString, redis::Jedis.Client)
        tokens = reduce(vcat, nltk_word_tokenize.(split_sentences(s)))
        words = unique([unique(tokens)..., lowercase.(unique(tokens))...])
        vectors = Jedis.hmget("vectors:word2vec", words...; client = redis)
        dictionary = Dict(words .=> vectors)
        vectors = stack(word.(tokens, Ref(dictionary)))
        vec(mean(vectors, dims = 2))
    end
    
end
