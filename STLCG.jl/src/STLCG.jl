module STLCG
using Revise
using EllipsisNotation
using LinearAlgebra
using Parameters # for @with_kw
using StatsFuns
using Zygote


export Maxish,
       Minish,
       TemporalFormula,
       Formula,
       Always,
       Eventually
       # LessThan,
       # Equal,
       # Negation,
       # Implies,
       # And,
       # Or,
       # Until,
       # Then,
       # Integral1D,
       # always,
       # eventually,
       # □,
       # ◊


##################################################
# Torch NN Modules
##################################################

# Note, anything after the ; indicates a keyword argument
# Remove this patch when https://github.com/FluxML/Zygote.jl/pull/793 is submitted
Zygote.collapse_nothings(xs::AbstractArray{Nothing}) = nothing
function Maxish(x; scale=0, dims=:, keepdims=true, distributed=false)
    if scale > 0
        maxish = logsumexp(x * scale; dims) / scale
    else
        m = maximum(x; dims)
        if distributed
            s = sum(x .* (x .== m); dims)
            maxish = s .* Zygote.dropgrad(m ./ s)
        else
            maxish = m
        end
    end
    keepdims && return maxish
    dims === (:) ? only(maxish) : dropdims(maxish; dims)
end


# Remove this patch when https://github.com/FluxML/Zygote.jl/pull/793 is submitted
function Minish(x; scale=0, dims=:, keepdims=true, distributed=false)
    if scale > 0
        minish = -logsumexp(-x * scale; dims) / scale
    else
        m = minimum(x; dims)
        if distributed
            s = sum(x .* (x .== m); dims)
            minish = s .* Zygote.dropgrad(m ./ s)
        else
            minish = m
        end
    end
    keepdims && return minish
    dims === (:) ? only(minish) : dropdims(minish; dims)
end


##################################################
# STL formulas
##################################################

abstract type Formula end
abstract type TemporalFormula end

@with_kw struct Always <: TemporalFormula
    subformula
    interval
    _interval = (interval == nothing) ? [0, Inf] : interval
    rnn_dim = (interval == nothing) ? Int8(1) : (interval[end] == Inf ? Int8(interval[1]) : Int8(interval[2]))
    steps = (interval == nothing) ? Int8(1) : Int8(diff([1,5])[1] + 1)
    operation = Minish
    M = begin
            dv = zeros(Int8(rnn_dim))
            ev = ones(Int8(rnn_dim-1))
            Array(Bidiagonal(dv, ev, :U))
        end
    b = begin
            bi = zeros(Int8(rnn_dim))
            bi[end] = 1.0
            bi
        end
end

@with_kw struct Eventually <: TemporalFormula
    subformula
    interval
    _interval = (interval == nothing) ? [0, Inf] : interval
    rnn_dim = (interval == nothing) ? Int8(1) : (interval[end] == Inf ? Int8(interval[1]) : Int8(interval[2]))
    steps = (interval == nothing) ? Int8(1) : Int8(diff([1,5])[1] + 1)
    operation = Maxish
    M = begin
            dv = zeros(Int8(rnn_dim))
            ev = ones(Int8(rnn_dim-1))
            Array(Bidiagonal(dv, ev, :U))
        end
    b = begin
            bi = zeros(Int8(rnn_dim))
            bi[end] = 1.0
            bi
        end
end

struct LessThan <: Formula
    lhs
    rhs
end

struct GreaterThan <: Formula
    lhs
    rhs
end

struct Equal <: Formula
    lhs
    rhs
end

struct Negation <: Formula
    subformula
end


@with_kw struct Implies <: Formula
    subformula1
    subformula2
    operation = Maxish
end


@with_kw struct And <: Formula
    subformula1
    subformula2
    operation = Minish
end


@with_kw struct Or <: Formula
    subformula1
    subformula2
    operation = Maxish
end

struct Until <: Formula
    subformula1
    subformula2

    # Until() = new(nothing, nothing) # example constructor
end

struct Then <: Formula
    subformula1
    subformula2

    # function Then()
    #     # example constructor (longer `function` format)
    #     return new(nothing, nothing)
    # end
end


# struct Integral1D <: Formula
#     subformula
#     padding_size
#     interval
#     conv
# end





function init_rnn_cell!(op::TemporalFormula, x)
    n = [size(x)...]
    # n[2] = op.rnn_dim
    h0 = ones(n...)[:,1:op.rnn_dim,..] .* x[:,1:1,..]
    ct = 0.0
    if (op._interval[2] == Inf) & (op._interval[1] > 0)
        d0 = x[:,1:1,..]
        init_h = ((d0, h0), ct)
    else
        init_h = (h0, ct)
    end
    return init_h

end

function rnn_cell(op::Always, x, hc; scale=0, distributed=false)
    h0, c = hc
    if op.interval == nothing
        if distributed
            new_h = (h0 .* c .+ x) .* (x .== h0) ./ (c .+ 1) + x .* (x .< h0) .+ h0 .* (x .> h0)
            new_c = (c .+ 1.0) .* (x .== h0) .+ 1.0 .* (x .< h0) .+ c .* (x .> h0)
            state = (new_h, new_c)
            output = new_h
        else
            # julia maximum function takes the first maximum value.
            # the order of x, h0 in the cat below matters
            input_ = cat(x, h0, dims=2)
            output = op.operation(input_; scale, distributed, dims=2)
            state = (output, nothing)
        end
    else
        # case: interval = [a, Inf)
        if (op._interval[2] == Inf) & (op._interval[1] > 0)
            d0, h00 = h0
            dh = cat(d0, h00[:,1:1,..], dims=2)
            output = op.operation(dh; scale, distributed, dims=2)
            new_h = cat(h00[:,2:end,:], x, dims=2)
            state = ((output, new_h), nothing)
        else
            state = (cat(h0[:,2:end,:], x, dims=2), nothing)
            input_ = cat(h0, x, dims=2)[:,1:op.steps,..]
            output = op.operation(input_; scale, distributed, dims=2)
        end
    end
    return (output, state)
end

function rnn_cell(op::Eventually, x, hc; scale=0, distributed=false)
    h0, c = hc
    if op.interval == nothing
        if distributed
            new_h = (h0 .* c .+ x) .* (x .== h0) ./ (c .+ 1) .+ x .* (x .> h0) .+ h0 .* (x .< h0)
            new_c = (c .+ 1.0) .* (x .== h0) .+ 1.0 .* (x .> h0) .+ c .* (x .< h0)
            state = (new_h, new_c)
            output = new_h
        else
            # julia maximum function takes the first maximum value.
            # the order of x, h0 in the cat below matters
            input_ = cat(x, h0, dims=2)
            output = op.operation(input_; scale, distributed, dims=2)
            state = (output, nothing)
        end
    else
        # case: interval = [a, Inf)
        if (op._interval[2] == Inf) & (op._interval[1] > 0)
            d0, h00 = h0
            dh = cat(d0, h00[:,1:1,..], dims=2)
            output = op.operation(dh; scale, distributed, dims=2)
            new_h = cat(h00[:,2:end,:], x, dims=2)
            state = ((output, new_h), nothing)
        else
            state = (cat(h0[:,2:end,:], x, dims=2), nothing)
            input_ = cat(h0, x, dims=2)[:,1:op.steps,..]
            output = op.operation(input_; scale, distributed, dims=2)
        end
    end
    return (output, state)
end





function run_rnn_cell(op::TemporalFormula, x; pscale=1, scale=0, keepdims=true, distributed=false)
    states = ()
    xx = robustness_trace(op.subformula, x; pscale, scale, keepdims, distributed)
    hc = init_rnn_cell!(op, xx)
    time_dim = size(xx)[2]
    for t in 1:time_dim
        o, hc = rnn_cell(op, xx[:,t:t,..], hc; scale, distributed)
        if t == 1
            global outputs = o
        else
            outputs = cat(outputs, o, dims=2)
        end
        states = (states..., hc)
    end
    return outputs, states
end


# function robustness_trace(op::TemporalFormula, trace; scale=0, dims=1, keepdims=True, distributed=false)
#     robustness_trace(op, x,)
# end



robustness_trace(formula::LessThan, trace; pscale=1.0, kwargs...) = (formula.rhs .- trace)*pscale
robustness_trace(formula::GreaterThan, trace; pscale=1.0, kwargs...) = (trace .- formula.rhs)*pscale
robustness_trace(formula::Equal, trace; pscale=1.0, kwargs...) = -abs.(trace .- formula.rhs)*pscale
robustness_trace(formula::Negation, trace; kwargs...) = -robustness_trace(formula.subformula, trace; kwargs...)

function robustness_trace(formula::Implies, trace; pscale=1, scale=0, keepdims=true, distributed=false, kwargs...)
    x, y = trace   # [bs, time, x_dim,...]
    trace1 = robustness_trace(formula.subformula1, x; pscale, scale, keepdims, distributed, kwargs...)
    trace2 = robustness_trace(formula.subformula2, y; pscale, scale, keepdims, distributed, kwargs...)
    xx = cat(-trace1, trace2, dims=length(size(x))+1)
    Maxish(xx; scale, dims=length(size(x))+1, keepdims, distributed)
end


function separate_and(formula, input; dims=4, pscale=1, scale=0, keepdims=true, distributed=false)
    if typeof(formula) != And
        return robustness_trace(formula, input; dims, pscale, scale, keepdims, distributed)
    else
        return cat(separate_and(formula.subformula1, input[1]; dims, pscale, scale, keepdims, distributed),
                   separate_and(formula.subformula2, input[2]; dims, pscale, scale, keepdims, distributed);
                   dims
                   )
    end
end

function robustness_trace(formula::And, trace; dims=4, pscale=1, scale=0, keepdims=true, distributed=false, kwargs...)
    xx = separate_and(formula, trace; dims, pscale, scale, keepdims, distributed)
    Minish(xx; scale, dims, keepdims=false, distributed)
end


function separate_or(formula, input; dims=4, pscale=1, scale=0, keepdims=true, distributed=false)
    if typeof(formula) != Or
        return robustness_trace(formula, input; dims, pscale, scale, keepdims, distributed)
    else
        return cat(separate_or(formula.subformula1, input[1]; dims, pscale, scale, keepdims, distributed),
                   separate_or(formula.subformula2, input[2]; dims, pscale, scale, keepdims, distributed);
                   dims
                   )
    end
end

function robustness_trace(formula::Or, trace; dims=4, pscale=1, scale=0, keepdims=true, distributed=false, kwargs...)
    xx = separate_or(formula, trace; dims, pscale, scale, keepdims, distributed)
    Maxish(xx; scale, dims, keepdims=false, distributed)
end

function robustness_trace(formula::TemporalFormula, trace; pscale=1, scale=0, keepdims=true, distributed=false, kwargs...)
    outputs, states = run_rnn_cell(formula, trace; pscale, scale, keepdims, distributed)
    outputs[:,end:end,..]
end


next_function(formula::TemporalFormula) = [formula.subformula]
next_function(formula::Union{LessThan, GreaterThan, Equal}) = [formula.lhs, formula.rhs]
next_function(formula::Negation) = [formula.subformula]
next_function(formula::Union{Implies, And, Or, Until, Then}) = [formula.subformula1, formula.subformula2]


□(subformula; interval=nothing) = Always(;subformula, interval)
◊(subformula; interval=nothing) = Eventually(;subformula, interval)

(op::Formula)(x; kwargs...) = robustness_trace(op, x; kwargs...)
(op::TemporalFormula)(x; kwargs...) = robustness_trace(op, x; kwargs...)

Base.print(op::STLCG.LessThan) = print(string(op.lhs) * " < " * string(op.rhs))
Base.print(op::STLCG.GreaterThan) = print(string(op.lhs) * " > " * string(op.rhs))
Base.print(op::STLCG.Equal) = print(string(op.lhs) * " = " * string(op.rhs))
Base.print(op::STLCG.Negation) = print("¬(" * string(op.subformula) *")")
Base.print(op::STLCG.Implies) = print(string(op.lhs) * " → " * string(op.rhs))
Base.print(op::STLCG.And) = print(string(op.lhs) * " ∧ " * string(op.rhs))
Base.print(op::STLCG.Or) = print(string(op.lhs) * " ∨ " * string(op.rhs))
Base.print(op::STLCG.Until) = print(string(op.lhs) * " U " * string(op.rhs))
Base.print(op::STLCG.Then) = print(string(op.lhs) * " T " * string(op.rhs))

# ##################################################
# # Old method of evaluating an STL formula below.
# ##################################################

# ⟶(p,q) = p ? q == true : true
# implies = → = ⇒ = ⟶ # aliases (\rightarrow, \Rightarrow)

# # \square □
# function always(predicate::Function, signals::AbstractArray, times::AbstractArray=0:length(signals)-1; interval=[0, Inf])
#     t0, t1 = interval
#     return all(implies(t0 <= t <= t1, predicate(s)) for (t,s) in zip(times, signals))
# end
# □(predicate::Function, signals) = always(predicate, signals) # without `times` and `interval`, assumes linear time b/w interval [0, Inf]
# □(predicate::Function, signals, times) = always(predicate, signals, times) # without interval, assumes [0, Inf]
# □(interval, predicate::Function, signals) = always(predicate, signals; interval=interval)  # without `times`, assumes linear time b/w interval
# □(interval, predicate::Function, signals, times) = always(predicate, signals, times; interval=interval)


# # \lozenge ◊
# function eventually(predicate::Function, signals::AbstractArray, times::AbstractArray=0:length(signals)-1; interval=[0, Inf])
#     t0, t1 = interval
#     return any(t0 <= t <= t1 ? predicate(s) : false for (t,s) in zip(times, signals))
# end
# ◊(predicate::Function, signals) = eventually(predicate, signals) # without `times` and `interval`, assumes linear time b/w interval [0, Inf]
# ◊(predicate::Function, signals, times) = eventually(predicate, signals, times) # without interval, assumes [0, Inf]
# ◊(interval, predicate::Function, signals) = eventually(predicate, signals; interval=interval)  # without `times`, assumes linear time b/w interval
# ◊(interval, predicate::Function, signals, times) = eventually(predicate, signals, times; interval=interval)


end # module
