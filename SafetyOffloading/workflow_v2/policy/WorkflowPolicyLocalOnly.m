function [action,X]=WorkflowPolicyLocalOnly(W)
%WORKFLOWPOLICYLOCALONLY BC weights but REQUEST_REMOTE_COMMIT forced off.
% Ablation: does forbidding remote commits raise TU on seeds 1301/1302?
c=WorkflowCandidateBuilder(W); X=WorkflowFreshCandidateFeatures(W,c);
score=bcScore(W,X);
if isempty(score)
    score=2*X(:,8)-0.15*X(:,6)-0.10*X(:,9)+0.05*X(:,10);
end
% Force off actions that add remote-commit latency
for i=1:numel(c)
    if c(i).kind=="REQUEST_REMOTE_COMMIT" || c(i).kind=="STOP_PLAN"
        score(i)=-1e6;
    end
end
% Same REJECT guard as the base policy
for i=1:numel(c)
    if c(i).kind~="REJECT_WORKFLOW", continue; end
    for j=1:numel(c)
        if c(j).kind=="COMMIT_LOCAL" && c(j).workflow_id==c(i).workflow_id
            score(i)=-1e6; break
        end
    end
end
[~,ix]=max(score);
action=c(ix); action.detail="local_only_greedy";

function score=bcScore(W,X)
score=[];
wfield='fresh_weights_json'; if ~isfield(W,wfield), wfield='fresh_bc_weights_json'; end
if ~isfield(W,wfield) || ~isfile(W.(wfield)), return; end
persistent cachePath cache
if isempty(cache) || ~strcmp(char(cachePath),char(W.(wfield)))
    d=jsondecode(fileread(W.(wfield))); s=d.state_dict;
    if isfield(d,'architecture') && string(d.architecture)=="dueling_ddqn"
        cache=struct('dueling',true, ...
            'ew',double(s.encoder_0_weight),'eb',double(s.encoder_0_bias(:))', ...
            'vw1',double(s.value_0_weight),'vb1',double(s.value_0_bias(:))', ...
            'vw2',double(s.value_2_weight),'vb2',double(s.value_2_bias), ...
            'aw1',double(s.advantage_0_weight),'ab1',double(s.advantage_0_bias(:))', ...
            'aw2',double(s.advantage_2_weight),'ab2',double(s.advantage_2_bias));
    else
        cache=struct('dueling',false,'w1',double(s.net_0_weight),'b1',double(s.net_0_bias(:))', ...
            'w2',double(s.net_2_weight),'b2',double(s.net_2_bias));
    end
    cachePath=W.(wfield);
end
if cache.dueling
    h=max(0,X*cache.ew'+cache.eb);
    z=mean(h,1);
    vh=max(0,z*cache.vw1'+cache.vb1); v=vh*cache.vw2'+cache.vb2;
    ah=max(0,[h repmat(z,size(h,1),1)]*cache.aw1'+cache.ab1);
    a=ah*cache.aw2'+cache.ab2;
    score=v+a-mean(a);
else
    h=max(0,X*cache.w1'+cache.b1);
    score=h*cache.w2'+cache.b2;
end
end
end
