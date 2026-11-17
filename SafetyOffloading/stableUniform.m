function u = stableUniform(seed, varargin)
%STABLEUNIFORM Deterministic U(0,1) keyed by immutable event identifiers.
% Uses an FNV-1a style hash represented with uint32 arithmetic.
s = string(seed);
for i = 1:numel(varargin)
    token = join(string(varargin{i}(:)), ",");
    s = s + "|" + token;
end
bytes = unicode2native(char(s), 'UTF-8');
h = uint32(2166136261);
for i = 1:numel(bytes)
    h = bitxor(h, uint32(bytes(i)));
    h = uint32(mod(uint64(h) * uint64(16777619), uint64(2)^32));
end
% Avalanche adjacent identifiers so neighboring vehicle IDs do not receive
% correlated SPS resources.
h = bitxor(h, bitshift(h,-16));
h = uint32(mod(uint64(h) * uint64(2246822507), uint64(2)^32));
h = bitxor(h, bitshift(h,-13));
h = uint32(mod(uint64(h) * uint64(3266489909), uint64(2)^32));
h = bitxor(h, bitshift(h,-16));
u = (double(h) + 0.5) / 2^32;
end

