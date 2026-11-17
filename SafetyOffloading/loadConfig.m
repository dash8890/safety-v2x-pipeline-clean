function cfg = loadConfig(path)
%LOADCONFIG Load and validate a safety-offloading JSON configuration.
arguments
    path (1,1) string
end
cfg = jsondecode(fileread(path));
required = {"project","radio","mobility","safety","compute","control","experiment"};
for i = 1:numel(required)
    assert(isfield(cfg, required{i}), "Config:MissingSection", ...
        "Missing configuration section: %s", required{i});
end
assert(cfg.radio.slot_duration_ms > 0, "Config:Slot", "slot_duration_ms must be positive");
assert(cfg.radio.subchannels >= 1, "Config:Subchannels", "subchannels must be positive");
assert(cfg.radio.max_transport_block_bytes >= 1, "Config:TB", "max transport block must be positive");
assert(cfg.safety.deadline_ms > 0 && cfg.compute.deadline_ms > 0, ...
    "Config:Deadline", "deadlines must be positive");
assert(cfg.compute.input_bytes > cfg.radio.max_transport_block_bytes, ...
    "Config:Fragmentation", "canonical input must exercise fragmentation");
end

