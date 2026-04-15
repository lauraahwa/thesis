function write_demand_response(path::AbstractString, inputs::Dict, setup::Dict, EP::Model)
    DR = inputs["DR"]
    if isempty(DR)
        return
    end

    gen          = inputs["RESOURCES"]
    T            = inputs["T"]
    omega        = inputs["omega"]
    scale_factor = setup["ParameterScale"] == 1 ? ModelScalingFactor : 1

    curtail = value.(EP[:vDR_CURTAIL]) * scale_factor

    rows = []
    for y in DR
        z     = zone_id(gen[y])
        rname = resource_name(gen[y])
        total_mwh  = sum(omega[t] * curtail[y, t] for t in 1:T)
        # inputs["pD"] set as [1,z] since data center has 1000 MW of flat load
        max_mw     = max_curtail_pct(gen[y]) * inputs["pD"][1, z] * scale_factor
        total_hours = max_mw > 0 ? sum(omega[t] * curtail[y, t] / max_mw for t in 1:T) : 0.0
        for t in 1:T
            demand_mw  = inputs["pD"][t, z] * scale_factor
            curtail_mw = curtail[y, t]
            pct = demand_mw > 0 ? 100.0 * curtail_mw / demand_mw : 0.0
            push!(rows, (
                Resource            = rname,
                Zone                = z,
                Total_Curtail_Hours = total_hours,
                Total_Curtail_MWh   = total_mwh,
                Timestep            = t,
                omega               = omega[t],
                vDR_CURTAIL_MW      = curtail_mw,
                Demand_MW           = demand_mw,
                Curtail_Pct         = pct,
            ))
        end
    end

    CSV.write(joinpath(path, "demand_response.csv"), DataFrame(rows))
end
