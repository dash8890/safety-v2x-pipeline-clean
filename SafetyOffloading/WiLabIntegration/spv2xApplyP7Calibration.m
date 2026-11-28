function aFeat=spv2xApplyP7Calibration(s,aFeat)
%SPV2XAPPLYP7CALIBRATION Apply shared P7 estimators to provisional assignment scores.
% No-op unless s.p7_models is populated (risk_myopic_p7). P6 remains untouched.
if ~isfield(s,'p7_models') || isempty(s.p7_models), return; end
m=s.p7_models;
if isfield(m,'action_risk') && ~isempty(m.action_risk)
    aFeat.predicted_conditional_safety_risk=calibrateActionRisk( ...
        m.action_risk,aFeat.predicted_conditional_safety_risk,aFeat.relative_distance_m);
end
if isfield(m,'completion') && ~isempty(m.completion)
    aFeat.predicted_completion_probability=calibrateCompletion(m.completion,aFeat);
end
end

function risk=calibrateActionRisk(model,provisional,distanceM)
risk=min(1,max(0,double(provisional)));
if ~isstruct(model), return; end
provisionalRisk=risk;
% Map provisional score through reliability bins (empirical upper = conservative).
if isfield(model,'calibration_bins')
    bins=model.calibration_bins;
    if iscell(bins), bins=[bins{:}]; end
    for i=1:numel(bins)
        b=bins(i);
        if provisionalRisk>=b.p_min && provisionalRisk<=b.p_max
            risk=double(b.empirical_upper);
            break
        end
    end
end
% Distance-bin empirical harm rates (jsondecode may use x1/x2 field names).
if isfield(model,'distance_bin_harm_rates') && isfinite(distanceM)
    bin=max(1,ceil(double(distanceM)/50));
    rates=model.distance_bin_harm_rates;
    fn=fieldnames(rates);
    for i=1:numel(fn)
        tok=regexp(fn{i},'(\d+)$','tokens','once');
        if ~isempty(tok) && str2double(tok{1})==bin
            risk=max(risk,double(rates.(fn{i})));
            break
        end
    end
end
if isfield(model,'policy_harm_floor')
    risk=max(risk,double(model.policy_harm_floor));
end
risk=min(1,max(0,risk));
end

function p=calibrateCompletion(model,aFeat)
% Start from provisional hard indicator / score already on the feature.
p=double(aFeat.predicted_completion_probability);
if ~isstruct(model) || ~isfield(model,'type') || model.type~="logistic"
    return
end
feats=string(model.features);
x=zeros(1,numel(feats));
for i=1:numel(feats)
    f=feats(i);
    switch f
        case "predicted_completion_probability", x(i)=double(aFeat.predicted_completion_probability);
        case "predicted_deadline_slack_ms"
            if isfield(aFeat,'predicted_remote_deadline_slack_ms')
                x(i)=double(aFeat.predicted_remote_deadline_slack_ms);
            elseif isfield(aFeat,'predicted_deadline_slack_ms')
                x(i)=double(aFeat.predicted_deadline_slack_ms);
            else
                x(i)=0;
            end
        case "predicted_conditional_safety_risk", x(i)=double(aFeat.predicted_conditional_safety_risk);
        case "processor_cpu_ghz", x(i)=double(aFeat.processor_cpu_ghz);
        case "is_local", x(i)=double(aFeat.is_local);
        case "is_local_action", x(i)=double(aFeat.is_local);
        case "relative_distance_m", x(i)=double(aFeat.relative_distance_m);
        otherwise, x(i)=0;
    end
end
means=double(model.means(:)');
scales=double(model.scales(:)');
scales(scales<1e-6)=1;
z=(x-means)./scales;
z(~isfinite(z))=0;
w=double(model.weights(:));
p=1/(1+exp(-(w(1)+sum(w(2:end).*z(:)))));
p=min(1,max(0,p));
end
