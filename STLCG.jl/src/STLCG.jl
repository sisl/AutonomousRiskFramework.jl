module STLCG

using Parameters # for @with_kw

export NNModule,
       Maxish,
       Minish,
       Operator,
       Formula,
       Always,
       Eventually,
       LessThan,
       Equal,
       Negation,
       Implies,
       And,
       Or,
       Until,
       Then,
       Integral1D,
       always,
       eventually,
       □,
       ◊


##################################################
# Torch NN Modules
##################################################

abstract type NNModule end

@with_kw mutable struct Maxish <: NNModule
    input_name::String = "Maxish input"
end


@with_kw mutable struct Minish <: NNModule
    input_name::String = "Minish input"
end

# Note, anything after the ; indicates a keyword argument
function forward(mod::Maxish, x, scale; dim=1, keepdim=true, agm=false, distributed=false)
    # ...
end

function forward(mod::Minish, x, scale; dim=1, keepdim=true, agm=false, distributed=false)
    # ...
end

next_function(mod::NNModule) = [mod.input_name]



##################################################
# STL operators and formulas
##################################################

abstract type Operator end
abstract type Formula end

mutable struct Always <: Operator
    subformula
    interval
    _interval
    rnn_dim
    steps
    operation
    M
    b
end


mutable struct Eventually <: Operator
    subformula
    interval
    _interval
    rnn_dim
    steps
    operation
    M
    b
end


mutable struct LessThan <: Formula
    lhs
    val
    subformula
end


mutable struct GreaterThan <: Formula
    lhs
    val
    subformula
end


mutable struct Equal <: Formula
    lhs
    val
    subformula
end


mutable struct Negation <: Formula
    subformula
end


@with_kw mutable struct Implies <: Formula
    subformula1
    subformula2
    operation = Maxish()
end


@with_kw mutable struct And <: Formula
    subformula1
    subformula2
    operation = Maxish()
end


@with_kw mutable struct Or <: Formula
    subformula1
    subformula2
    operation = Maxish()
end


mutable struct Until <: Formula
    subformula1
    subformula2

    Until() = new(nothing, nothing) # example constructor
end


mutable struct Then <: Formula
    subformula1
    subformula2

    function Then()
        # example constructor (longer `function` format)
        return new(nothing, nothing)
    end
end


mutable struct Integral1D <: Formula
    subformula
    padding_size
    interval
    conv
end


function init_rnn_cell!(op::Operator)
    # Catch-all for non-specialized Operator type
end


function init_rnn_cell!(op::Eventually)
    # ...
end


function init_rnn_cell!(op::LessThan)
    # ...
end


function run_rnn_cell(op::Operator)
    # Catch-all for non-specialized Operator type
end


function rnn_cell(op::Always)
    # ...
end


function robustness_trace(formula::Formula)
    # Catch-all
end


robustness_trace(formula::LessThan, trace, pscale=1.0) = (formula.val - trace)*pscale
robustness_trace(formula::GreaterThan, trace, pscale=1.0) = (trace - formula.val)*pscale


next_function(formula::Formula) = [formula.lhs, formula.val]
next_function(formula::Union{Negation, Integral1D}) = [formula.subformula]
next_function(formula::Union{Implies, And, Or, Until, Then}) = [formula.subformula1, formula.subformula2]



##################################################
# Old method of evaluating an STL formula below.
##################################################

⟶(p,q) = p ? q == true : true
implies = → = ⇒ = ⟶ # aliases (\rightarrow, \Rightarrow)

# \square □
function always(predicate::Function, signals::AbstractArray, times::AbstractArray=0:length(signals)-1; interval=[0, Inf])
    t0, t1 = interval
    return all(implies(t0 <= t <= t1, predicate(s)) for (t,s) in zip(times, signals))
end
□(predicate::Function, signals) = always(predicate, signals) # without `times` and `interval`, assumes linear time b/w interval [0, Inf]
□(predicate::Function, signals, times) = always(predicate, signals, times) # without interval, assumes [0, Inf]
□(interval, predicate::Function, signals) = always(predicate, signals; interval=interval)  # without `times`, assumes linear time b/w interval
□(interval, predicate::Function, signals, times) = always(predicate, signals, times; interval=interval)


# \lozenge ◊
function eventually(predicate::Function, signals::AbstractArray, times::AbstractArray=0:length(signals)-1; interval=[0, Inf])
    t0, t1 = interval
    return any(t0 <= t <= t1 ? predicate(s) : false for (t,s) in zip(times, signals))
end
◊(predicate::Function, signals) = eventually(predicate, signals) # without `times` and `interval`, assumes linear time b/w interval [0, Inf]
◊(predicate::Function, signals, times) = eventually(predicate, signals, times) # without interval, assumes [0, Inf]
◊(interval, predicate::Function, signals) = eventually(predicate, signals; interval=interval)  # without `times`, assumes linear time b/w interval
◊(interval, predicate::Function, signals, times) = eventually(predicate, signals, times; interval=interval)


end # module
