
struct Occurrences
    bon
    labels
    time
end
Base.length(pres::Occurrences) = length(pres.bon)
Base.show(io::IO, pres::Occurrences) = print(io, "$(length(pres)) occurrence records from time $(pres.time)")
aggregate_coordinates(occs::Vector{<:Occurrences}) = vcat([occs[i].coords for i in eachindex(occs)]...)


struct OccurrenceTimeseries
    occurrences::Vector{Occurrences}
end
Base.show(io::IO, pt::OccurrenceTimeseries) = print(io, "Presence timeseries with $(length(pt.occurrences)) timepoints")
Base.getindex(pt::OccurrenceTimeseries, i) = pt.occurrences[i]
Base.eachindex(ot::OccurrenceTimeseries) = eachindex(ot.occurrences)
aggregate_coordinates(ot::OccurrenceTimeseries) = vcat([ot[i].coords for i in eachindex(ot)]...)


function sample_occurrence(
    sr::ShiftingRange,
    bon::BiodiversityObservationNetwork;
    t = 1
)
    _range = sr.scores[t] .> sr.threshold
    labels = [_range[n...] for n in bon.nodes]
    return Occurrences(bon, labels, t)
end


function sample_occurrence(
    sr::ShiftingRange,
    bon::BiodiversityObservationNetwork,
    n::Vector{<:Int}
)
    T = length(n)
    OccurrenceTimeseries([sample(sr, bon; t=t) for t in 1:T])
end