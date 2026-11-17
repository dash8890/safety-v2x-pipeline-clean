function [x,y] = vehiclePositions(trace, cfg, time_ms)
%VEHICLEPOSITIONS Position vehicles on the wrapped six-lane highway.
x = mod(trace.x0_m + trace.direction .* trace.speed_mps .* (time_ms/1000), ...
    cfg.mobility.road_length_m);
y = (trace.lane - 0.5) * cfg.mobility.lane_width_m;
end

