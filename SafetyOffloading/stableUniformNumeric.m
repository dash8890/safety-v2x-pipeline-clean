function u=stableUniformNumeric(seed,tag,varargin)
%STABLEUNIFORMNUMERIC Fast deterministic U(0,1) for numeric event keys.
h=uint32(mod(double(seed),2^32));
h=mix(h,uint32(tag));
for i=1:numel(varargin)
    value=varargin{i};
    assert(isscalar(value),'StableUniformNumeric:Scalar','Numeric hash keys must be scalar');
    scaled=round(double(value));
    h=mix(h,uint32(mod(scaled,2^32)));
end
h=bitxor(h,bitshift(h,-16)); h=uint32(mod(uint64(h)*uint64(2246822507),uint64(2)^32));
h=bitxor(h,bitshift(h,-13)); h=uint32(mod(uint64(h)*uint64(3266489909),uint64(2)^32));
h=bitxor(h,bitshift(h,-16)); u=(double(h)+0.5)/2^32;
end

function h=mix(h,value)
h=bitxor(h,value+uint32(2654435769));
h=uint32(mod(uint64(h)*uint64(16777619),uint64(2)^32));
end
