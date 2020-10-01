module STLCG

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


mutable struct LessThan <: Formula
    lhs
    rhs
end


mutable struct GreaterThan <: Formula
    lhs
    rhs
end


mutable struct Equal <: Formula
    lhs
    rhs
end


mutable struct Negation <: Formula
    subformula
end


@with_kw mutable struct Implies <: Formula
    subformula1
    subformula2
    operation = Maxish
end


@with_kw mutable struct And <: Formula
    subformula1
    subformula2
    operation = Minish
end


@with_kw mutable struct Or <: Formula
    subformula1
    subformula2
    operation = Maxish
end


mutable struct Until <: Formula
    subformula1
    subformula2

    # Until() = new(nothing, nothing) # example constructor
end


mutable struct Then <: Formula
    subformula1
    subformula2

    # function Then()
    #     # example constructor (longer `function` format)
    #     return new(nothing, nothing)
    # end
end


# mutable struct Integral1D <: Formula
#     subformula
#     padding_size
#     interval
#     conv
# end


# function init_rnn_cell!(op::Operator)
#     # Catch-all for non-specialized Operator type
# end


function init_rnn_cell!(op::TemporalFormula, x)
    n = [size(x)...]
    n[2] = op.rnn_dim
    h0 = ones(n...) .* x[:,1:1,..]
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
            new_h = (h0 * c + x) .* (x .== h0) / (c + 1) + x .* (x .< h0) + h0 .* (x .> h0)
            new_c = (c + 1.0) .* (x .== h0) + 1.0 .* (x .< h0) + c .* (x .> h0)
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
            new_h = cat(h00[:,1:end-1,:], x, dims=2)
            state = ((output, new_h), nothing)
        else
            state = cat(h0[:,1:end-1,:], x, dims=2)
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
            new_h = (h0 * c + x) .* (x .== h0) / (c + 1) + x .* (x .< h0) + h0 .* (x .> h0)
            new_c = (c + 1.0) .* (x .== h0) + 1.0 .* (x .< h0) + c .* (x .> h0)
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
            new_h = cat(h00[:,1:end-1,:], x, dims=2)
            state = ((output, new_h), nothing)
        else
            state = cat(h0[:,1:end-1,:], x, dims=2)
            input_ = cat(h0, x, dims=2)[:,1:op.steps,..]
            output = op.operation(input_; scale, distributed, dims=2)
        end
    end
    return output, states
end



function run_rnn_cell(op::TemporalFormula, x; scale=0, distributed=false)
    outputs = []
    states = []
    hc = init_rnn_cell!(op, x)
    time_dim = size(x)[2]
    xs = [x[:,i:i,..] for i in 1:time_dim]
    for t in 1:time_dim
        o, hc = rnn_cell(op, xs[t], hc)
        push!(outputs, o)
        push!(states, hc)
    end
    return outputs, states
end




# function robustness_trace(formula::Formula)
#     # Catch-all
# end


# robustness_trace(formula::LessThan, trace, pscale=1.0) = (formula.val - trace)*pscale
# robustness_trace(formula::GreaterThan, trace, pscale=1.0) = (trace - formula.val)*pscale


# next_function(formula::Formula) = [formula.lhs, formula.val]
# next_function(formula::Union{Negation, Integral1D}) = [formula.subformula]
# next_function(formula::Union{Implies, And, Or, Until, Then}) = [formula.subformula1, formula.subformula2]



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
