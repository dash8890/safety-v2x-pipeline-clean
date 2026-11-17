function trace = generateScenario(cfg, density, seed)
%GENERATESCENARIO Generate paired-world exogenous mobility and task releases.
rng(seed, "twister");
N = max(2, round(density * cfg.mobility.road_length_m / 1000));
idx = find(cfg.mobility.density_vehicles_per_km == density, 1);
if isempty(idx), idx = 1; end
mu = cfg.mobility.speed_kmh_mean(min(idx, numel(cfg.mobility.speed_kmh_mean)));
sigma = cfg.mobility.speed_kmh_std(min(idx, numel(cfg.mobility.speed_kmh_std)));
trace.seed = seed;
trace.density = density;
trace.vehicle_id = (1:N)';
trace.x0_m = rand(N,1) * cfg.mobility.road_length_m;
trace.lane = mod((0:N-1)', cfg.mobility.lane_count) + 1;
trace.direction = ones(N,1); trace.direction(trace.lane > cfg.mobility.lane_count/2) = -1;
trace.speed_mps = max(1, (mu + sigma * randn(N,1)) / 3.6);
lo = cfg.compute.cpu_cycles_per_s_min; hi = cfg.compute.cpu_cycles_per_s_max;
trace.cpu_rate = lo + (hi-lo) * rand(N,1);

originCount = max(1, round(N * cfg.compute.origin_fraction));
origins = trace.vehicle_id(randperm(N, originCount));
release = (0:cfg.compute.arrival_interval_ms:(cfg.experiment.duration_s*1000-1))';
M = numel(release) * originCount;
trace.tasks = table('Size',[M 9], ...
    'VariableTypes',{'double','double','double','double','double','double','double','double','double'}, ...
    'VariableNames',{'task_id','origin_id','release_ms','deadline_ms','input_bytes','output_bytes','cycles','value','jitter_key'});
k = 0;
for r = 1:numel(release)
    for o = 1:originCount
        k = k + 1;
        trace.tasks{k,:} = [k origins(o) release(r) release(r)+cfg.compute.deadline_ms, ...
            cfg.compute.input_bytes cfg.compute.output_bytes cfg.compute.cycles_per_task 1 ...
            stableUniform(seed,"task",k)];
    end
end
end

