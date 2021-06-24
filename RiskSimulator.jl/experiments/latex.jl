
function latex_risk_table(planners)
    systems = ["IntelligentDriverModel", "PrincetonDriver", "MPCDriver"]
    scenarios = [CROSSING, T_HEAD_ON, T_LEFT, STOPPING, MERGING, CROSSWALK]
    seeds = 1:5
    α = 0.2

    multirow = s->"\\multirow{3}{*}{$s}"
    format_scenario = Dict(
        CROSSING=>"Crossing intersection",
        T_HEAD_ON=>"Head-on T-turn",
        T_LEFT=>"Left T-turn",
        STOPPING=>"Highway stopping",
        MERGING=>"Highway merging",
        CROSSWALK=>"Pedestrian crosswalk",
        "OVERALL"=>"Overall")
    format_system = Dict("IntelligentDriverModel"=>"IDM   ", "PrincetonDriver"=>"Princeton   ", "MPCDriver"=>"MPC Behavior")
    padding = x->string(x)[2]=='.' ? x : rpad(x, 5, "0")
    rounding = x->padding(round(x, digits=2))
    pm = (μ,σ)->format(string(rounding(μ), "_{\\pm ", rounding(σ), "}"))
    format = x->string("\$", x, "\$")

    # Table header:
    # Scenario & AV Policy & $\mathbb{E}[Z]$ & VaR & CVaR & Worst-Case & Failure Rate & FFE & $\max\log(p)$ & "Risk AUC" \\

    risk_metrics_for_plotting = []
    ast_metrics_for_plotting = []
    datasets_for_plotting = []
    failure_metrics_for_plotting = []

    function show_table_row(local_planners, system, scenario; first=true, combined_distributions=false)
        fail_metrics_vector = failure_metrics.(local_planners, verbose=false)
        fail_metrics_mean = mean(fail_metrics_vector)
        fail_metrics_std = std(fail_metrics_vector)

        if combined_distributions
            risk_metrics = collect_metrics(local_planners, α)
            risk_rows = string(
                format(rounding(risk_metrics.mean)), " \t& ",
                format(rounding(risk_metrics.var)), "\t& ",
                format(rounding(risk_metrics.cvar)), " \t& ",
                format(rounding(risk_metrics.worst)), "\t& ")
        else
            risk_metrics_vector = risk_assessment.(local_planners, α)
            risk_metrics_mean = mean(risk_metrics_vector)
            risk_metrics_std = std(risk_metrics_vector)
            risk_rows = string(
                pm(risk_metrics_mean.mean, risk_metrics_std.mean), " \t& ",
                pm(risk_metrics_mean.var, risk_metrics_std.var), " \t& ",
                pm(risk_metrics_mean.cvar, risk_metrics_std.cvar), " \t& ",
                pm(risk_metrics_mean.worst, risk_metrics_std.worst), " \t& ",
                )
        end

        # Risk AUC.
        𝐰 = ones(7)
        𝐰[end-2] = 10
        𝐰[end-1] = 10
        𝐰[end] = 10
        # 𝐰[end] = inverse_max_likelihood([failure_metrics_vector, failure_metrics_vector2])
        # 𝐰[end] = inverse_max_likelihood([fail_metrics_vector])
        dataset = combine_datasets(local_planners)
        ast_metrics = combine_ast_metrics(local_planners)
        areas = overall_area([dataset], [ast_metrics], weights=𝐰, α=α)
        risk_auc = round(areas[1], digits=5)

        mr = multirow(format_scenario[scenario])
        scenario_label = first ?  mr : " "^length(mr)
        println(scenario_label, "  \t& ",
                format_system[system], "\t& ",
                risk_rows,
                pm(fail_metrics_mean.failure_rate, fail_metrics_std.failure_rate), " \t& ",
                pm(fail_metrics_mean.first_failure, fail_metrics_std.first_failure), "\t& ",
                pm(fail_metrics_mean.highest_loglikelihood, fail_metrics_std.highest_loglikelihood), " \t& ",
                format(rounding(risk_auc)), " \\\\")

        push!(risk_metrics_for_plotting, risk_metrics_mean)
        push!(ast_metrics_for_plotting, ast_metrics)
        push!(datasets_for_plotting, dataset)
        push!(failure_metrics_for_plotting, fail_metrics_vector)
        
        if system == systems[end]
            if scenario == "OVERALL"
                scenario_string = "Overall"
                println("\\bottomrule")
            elseif scenario == "CROSSWALK"
                scenario_string = get_scenario_string(scenario)
                println("\\midrule")
            else
                scenario_string = get_scenario_string(scenario)
                println("\\greyrule")
            end

            if scenario in [T_LEFT]
            # if scenario in [CROSSING, T_HEAD_ON, T_LEFT, STOPPING, MERGING, CROSSWALK, "OVERALL"] # missing T_LEFT
                @info "Plotting..."

                # pcc = plot_combined_cost(risk_metrics_for_plotting, ["IDM", "Princeton", "MPC Behavior"])
                # RiskSimulator.savefig(pcc, "cost-$(lowercase(string(scenario))).tex")

                for equal in [false, true]
                    file_label = lowercase(string(scenario))
                    𝐰 = ones(7)

                    if !equal
                        𝐰[end-2] = 𝐰[end-1] = 𝐰[end] = 10
                        # 𝐰[end] = inverse_max_likelihood(failure_metrics_for_plotting) 
                    end

                    if equal && scenario == "OVERALL"
                        file_label = string(file_label, "-equal-weights")
                        scenario_string = string(scenario_string, " (uniform weights)")
                    end

                    areas = overall_area(datasets_for_plotting, ast_metrics_for_plotting, weights=𝐰, α=α)
                    area_idm = round(areas[1], digits=5)
                    area_princeton = round(areas[2], digits=5)
                    area_mpc = round(areas[3], digits=5)
                    p_metrics = plot_polar_risk(datasets_for_plotting, ast_metrics_for_plotting,
                        ["IDM ($area_idm)", "Princeton ($area_princeton)", "MPC Behavior ($area_mpc)"]; 
                        weights=𝐰, α=α, title="Risk area: $scenario_string")
                    RiskSimulator.Plots.PyPlot.savefig("polar-$file_label.pdf", bbox_inches="tight")

                    if scenario != "OVERALL"
                        break
                    end
                end
            end
            risk_metrics_for_plotting = [] # clear
            ast_metrics_for_plotting = [] # clear
            datasets_for_plotting = [] # clear
            failure_metrics_for_plotting = [] # clear
        end
    end



    for scenario in scenarios

        for system in systems
            local_planners = []
            for seed in seeds
                push!(local_planners, planners[join([system, scenario, "seed$seed"], ".")])
            end
            show_table_row(local_planners, system, scenario, first=(system==systems[1]))
        end

        # Save figures
        # pcc = plot_combined_cost([metrics, metrics2], ["IDM", "Princeton", "MPC Behavior"]; mean_y=3.33, var_y=3.25, cvar_y=2.1, α_y=2.8)
    end

    # Overall.
    for system in systems
        overall_planners = []
        for scenario in scenarios
            for seed in seeds
                push!(overall_planners, planners[join([system, scenario, "seed$seed"], ".")])
            end
        end
        show_table_row(overall_planners, system, "OVERALL", first=(system==systems[1]))
    end
end