function [obs,mask,st]=WorkflowCausalObs(W)
%WORKFLOWCAUSALOBS 18-D causal observation + 5-action mask.
% Features 17/18 close the former admission-state aliasing: the congest
% teacher depends on held-edge count and minimum active high-value slack.
obs=zeros(18,1);
mask=true(5,1);
st=pickEligibleEDF(W);
if isempty(st)
    mask(:)=false; mask(1)=true; % LOCAL noop
    return
end
deadlineS=0.35; if isfield(W,'rel_deadline_s'), deadlineS=double(W.rel_deadline_s); end
horizonS=1.45; if isfield(W,'horizon_s'), horizonS=double(W.horizon_s); end
vh=12; if isfield(W,'value_high'), vh=double(W.value_high); end
nArr=20; if isfield(W,'arrival_plan'), nArr=max(numel(W.arrival_plan),1); end

qnorm=zeros(3,1);
for v=1:min(3,numel(W.cpu))
    q=queuedCycles(W.cpu(v));
    hz=max(W.cpu(v).available_hz,1);
    qnorm(v)=q/(hz*deadlineS+1e-9);
end
c2=0; c3=0;
if size(W.contact,1)>=2, c2=double(W.contact(W.origin_id,2)); end
if size(W.contact,1)>=3, c3=double(W.contact(W.origin_id,3)); end

wf=wfOf(W,st.workflow_id);
slack=(wf.absolute_deadline_s-W.now_s)/deadlineS;
stype=stageType(st.stage_id);
cyclesN=st.required_cycles/12e6;
nActive=countStatus(W,"ACTIVE")/nArr;
f10=contactAgeNorm(W,2,horizonS);
f11=advAgeNorm(W,2,horizonS);
nRemote=countRemoteDone(W)/max(nArr*2,1);
nSucc=countStatus(W,"SUCCEEDED")/nArr;
valN=wf.value/vh;
nElig=countEligible(W)/8.0;
heldN=countHeld(W)/8.0;
minHighSlackN=minHighSlack(W,vh,deadlineS);

obs=[qnorm(1);qnorm(2);qnorm(3);c2;c3; ...
    max(min(slack,2),-1);stype;cyclesN;nActive;W.now_s/horizonS; ...
    f10;f11;nRemote;nSucc;valN;max(min(nElig,1),0); ...
    max(min(heldN,1),0);minHighSlackN];

% Action mask: A is the admission decision; D must complete locally.  DROP at
% D is strictly dominated because A/B/C costs are already sunk and completion
% has non-negative value.  B/C expose placement and hold actions.
mask(:)=true;
if st.stage_id=="A"
    mask(:)=false; mask(1)=true; mask(5)=true; % LOCAL, DROP
elseif st.stage_id=="D"
    mask(:)=false; mask(1)=true; % complete the admitted workflow
else
    mask(5)=false; % admission/drop is decided once, at stage A
    if size(W.contact,1)<2 || ~W.contact(W.origin_id,2), mask(2)=false; end
    if size(W.contact,1)<3 || ~W.contact(W.origin_id,3), mask(3)=false; end
end
end

function st=pickEligibleEDF(W)
st=[];
best=inf;
for i=1:numel(W.stages)
    s=W.stages(i);
    if s.status~="ELIGIBLE", continue; end
    wf=wfOf(W,s.workflow_id);
    key=wf.absolute_deadline_s+1e-9*double(s.workflow_id);
    if key<best, best=key; st=s; end
end
end

function q=queuedCycles(cpu)
q=0;
if ~isempty(cpu.running), q=q+cpu.running.remaining; end
if ~isempty(cpu.queue)
    for j=1:numel(cpu.queue), q=q+cpu.queue(j).remaining; end
end
end

function n=countStatus(W,st)
n=0;
for i=1:numel(W.workflows)
    if W.workflows(i).status==string(st), n=n+1; end
end
end

function n=countEligible(W)
n=0;
for i=1:numel(W.stages)
    if W.stages(i).status=="ELIGIBLE", n=n+1; end
end
end

function n=countRemoteDone(W)
n=0;
for i=1:numel(W.stages)
    if W.stages(i).status=="DONE" && W.stages(i).placement_id>0 && ...
            W.stages(i).placement_id~=W.origin_id
        n=n+1;
    end
end
end

function n=countHeld(W)
n=0;
for i=1:numel(W.edges)
    if W.edges(i).status=="READY_HELD" || W.edges(i).status=="WAIT_ADMISSION"
        n=n+1;
    end
end
end

function s=minHighSlack(W,vh,deadlineS)
s=2;
found=false;
for i=1:numel(W.workflows)
    wf=W.workflows(i);
    if wf.status=="ACTIVE" && wf.value>=vh-1e-9
        s=min(s,(wf.absolute_deadline_s-W.now_s)/max(deadlineS,1e-9));
        found=true;
    end
end
if ~found, s=2; end
s=max(min(s,2),-1);
end

function t=stageType(sid)
switch string(sid)
    case "A", t=0.0;
    case "B", t=0.33;
    case "C", t=0.66;
    otherwise, t=1.0;
end
end

function a=contactAgeNorm(W,helper,horizonS)
if helper>size(W.contact,1) || ~W.contact(W.origin_id,helper), a=0; return; end
t0=0;
if isfield(W,'last_contact_change_t') && numel(W.last_contact_change_t)>=helper
    t0=W.last_contact_change_t(helper);
end
a=max(min((W.now_s-t0)/horizonS,1),0);
end

function a=advAgeNorm(W,helper,horizonS)
% Causal advertisement age; fall back to contact-up age / TTL proxy.
ttl=0.30;
if isfield(W,'advertisement_ttl_s'), ttl=double(W.advertisement_ttl_s); end
if isfield(W,'last_adv_rx_t') && numel(W.last_adv_rx_t)>=helper && isfinite(W.last_adv_rx_t(helper))
    age=W.now_s-W.last_adv_rx_t(helper);
    a=max(min(age/horizonS,1),0);
    return
end
if helper<=size(W.contact,1) && W.contact(W.origin_id,helper)
    a=contactAgeNorm(W,helper,horizonS);
else
    a=max(min(ttl/horizonS,1),0);
end
end

function wf=wfOf(W,id)
wf=W.workflows(1);
for i=1:numel(W.workflows)
    if W.workflows(i).workflow_id==id, wf=W.workflows(i); return; end
end
end
