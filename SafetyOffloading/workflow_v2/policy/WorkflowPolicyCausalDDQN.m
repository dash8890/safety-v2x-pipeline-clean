function acts=WorkflowPolicyCausalDDQN(W)
% Outcome A closeout policy:
%  - Admission: hard DROP-low + congest shield (NOT learned)
%  - Placement: DAgger mix — with prob beta follow congest teacher placement,
%    else greedy DDQN over {LOCAL,H2,H3,HOLD}. No ε/Boltzmann suicide.
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
[obs,mask,st]=WorkflowCausalObs(W);
if isempty(st)
    acts=releaseReadyEdges(W);
    return
end
vh=12; if isfield(W,'value_high'), vh=double(W.value_high); end
wf=wfOf(W,st.workflow_id);
eps=0; if isfield(W,'ddqn_eps'), eps=double(W.ddqn_eps); end %#ok<NASGU>
beta=0; if isfield(W,'ddqn_dagger_beta'), beta=double(W.ddqn_dagger_beta); end

forceDrop=false;
dropWhy="ddqn_drop";
if st.stage_id=="A" && wf.value<vh-1e-9 && mask(5)
    forceDrop=true;
    dropWhy="ddqn_drop_low_hard";
    mask(:)=false; mask(5)=true;
elseif st.stage_id=="A" && wf.value>=vh-1e-9 && shouldBlockHigh(W,vh)
    forceDrop=true;
    dropWhy="ddqn_drop_admit_shield";
    mask(:)=false; mask(5)=true;
end

if forceDrop
    aIdx=5;
elseif beta>0 && rand()<beta
    aIdx=teacherPlacement(W,st,mask);
else
    aIdx=duelingGreedyPlace(W,obs,mask,st,wf,vh);
end

ddqnLastSet(struct('obs',obs(:)','mask',double(mask(:)'),'action_idx',aIdx, ...
    'workflow_id',st.workflow_id,'stage_id',string(st.stage_id)));
switch aIdx
    case 1, acts(1)=mk("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,"","ddqn_local");
    case 2, acts(1)=mk("REQUEST_REMOTE",st.workflow_id,st.stage_id,2,"","ddqn_h2");
    case 3, acts(1)=mk("REQUEST_REMOTE",st.workflow_id,st.stage_id,3,"","ddqn_h3");
    case 4, acts(1)=mk("STOP",st.workflow_id,st.stage_id,0,"","ddqn_hold");
    case 5, acts(1)=mk("DROP",st.workflow_id,st.stage_id,0,"",dropWhy);
    otherwise, acts(1)=mk("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,"","ddqn_fallback");
end
end

function aIdx=duelingGreedyPlace(W,obs,mask,st,wf,vh)
% Greedy placement only — DROP banned unless admission already forced it.
legal=find(mask(:));
if isempty(legal), aIdx=1; return; end
w=getWeights(W);
h=max(0,w.fc1w*obs(:)+w.fc1b);
h=max(0,w.fc2w*h+w.fc2b);
v=w.vw*h+w.vb; adv=w.aw*h+w.ab;
q=v+adv-mean(adv); q(~mask)=-1e9;
q(5)=-1e9; % placement-only learning
if ~any(q>-1e8)
    if mask(1), aIdx=1; else, aIdx=legal(1); end
    return
end
[~,aIdx]=max(q);
end

function aIdx=teacherPlacement(W,st,mask)
% Congest teacher placement: A/D local; B/C mobility EFT among legal.
if st.stage_id=="A" || st.stage_id=="D"
    if mask(1), aIdx=1; return; end
end
[hid,~]=pickMobile(W,st);
if hid==W.origin_id && mask(1)
    aIdx=1; return
end
if hid==2 && mask(2), aIdx=2; return; end
if hid==3 && mask(3), aIdx=3; return; end
if mask(1), aIdx=1; return; end
if mask(4), aIdx=4; return; end
legal=find(mask(:));
aIdx=legal(1);
end

function [hid,why]=pickMobile(W,st)
best=inf; hid=W.origin_id; why="local_mob";
for v=1:numel(W.cpu)
    if v~=W.origin_id && ~W.contact(W.origin_id,v), continue; end
    q=0;
    if ~isempty(W.cpu(v).running), q=q+W.cpu(v).running.remaining; end
    if ~isempty(W.cpu(v).queue)
        for j=1:numel(W.cpu(v).queue), q=q+W.cpu(v).queue(j).remaining; end
    end
    t=q/max(W.cpu(v).available_hz,1)+st.required_cycles/max(W.cpu(v).available_hz,1);
    if v~=W.origin_id, t=t+2*max(W.default_link_delay_s,0.010); end
    if t<best, best=t; hid=v; why=sprintf('mob_v%d',v); end
end
end

function tf=shouldBlockHigh(W,vh)
nH=countActiveHigh(W,vh);
if nH<3, tf=false; return; end
held=countHeldEdges(W);
minSlack=minActiveHighSlack(W,vh);
tf = (nH>=3 && held>=4 && minSlack<0.10) || (nH>=6);
end

function n=countActiveHigh(W,vh)
n=0;
for i=1:numel(W.workflows)
    wf=W.workflows(i);
    if wf.status=="ACTIVE" && wf.value>=vh-1e-9, n=n+1; end
end
end

function n=countHeldEdges(W)
n=0;
for i=1:numel(W.edges)
    ed=W.edges(i);
    if ed.status=="READY_HELD" || ed.status=="WAIT_ADMISSION", n=n+1; end
end
end

function s=minActiveHighSlack(W,vh)
s=inf;
for i=1:numel(W.workflows)
    wf=W.workflows(i);
    if wf.status=="ACTIVE" && wf.value>=vh-1e-9
        s=min(s, wf.absolute_deadline_s-W.now_s);
    end
end
if ~isfinite(s), s=inf; end
end

function wf=wfOf(W,id)
wf=W.workflows(1);
for i=1:numel(W.workflows)
    if W.workflows(i).workflow_id==id, wf=W.workflows(i); return; end
end
end

function ddqnLastSet(s)
global DDQN_LAST_DECISION
DDQN_LAST_DECISION=s;
end

function w=getWeights(W)
persistent Wcache Wpath
want=""; if isfield(W,'ddqn_weights_json'), want=string(W.ddqn_weights_json); end
if isempty(Wcache) || (strlength(want)>0 && (isempty(Wpath) || Wpath~=want))
    Wcache=loadWeights(W); Wpath=want;
end
w=Wcache;
end

function w=loadWeights(W)
path=""; if isfield(W,'ddqn_weights_json'), path=string(W.ddqn_weights_json); end
if strlength(path)==0 || ~isfile(path)
    root=fileparts(fileparts(mfilename('fullpath')));
    path=fullfile(root,'drl','outputs','ddqn_wilab_congest.weights.json');
end
assert(isfile(path),'DDQN weights missing: %s',path);
ww=jsondecode(fileread(path)).weights;
w=struct('fc1w',asMat(ww.fc1_weight),'fc1b',asVec(ww.fc1_bias), ...
    'fc2w',asMat(ww.fc2_weight),'fc2b',asVec(ww.fc2_bias), ...
    'vw',asMat(ww.v_weight),'vb',asVec(ww.v_bias), ...
    'aw',asMat(ww.a_weight),'ab',asVec(ww.a_bias));
end

function M=asMat(x)
if isnumeric(x), M=double(x); return; end
if iscell(x)
    M=zeros(numel(x),numel(x{1}));
    for i=1:numel(x)
        r=x{i}; if iscell(r), M(i,:)=cellfun(@double,r); else, M(i,:)=double(r); end
    end
    return
end
M=double(x);
end

function v=asVec(x)
if isnumeric(x), v=double(x(:)); return; end
if iscell(x), v=cellfun(@double,x); v=v(:); return; end
v=double(x(:));
end

function acts=releaseReadyEdges(W)
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
for i=1:numel(W.edges)
    ed=W.edges(i);
    if ed.status~="READY_HELD", continue; end
    if ed.src_vehicle<=0 || ed.dst_vehicle<=0 || ed.src_vehicle==ed.dst_vehicle, continue; end
    if W.contact(ed.src_vehicle,ed.dst_vehicle)
        acts(end+1)=mk("RELEASE_EDGE",ed.workflow_id,"",0,ed.name,"contact_ok"); %#ok<AGROW>
    end
end
end

function a=mk(type,wf,sid,hid,ename,detail)
a=struct('type',string(type),'workflow_id',wf,'stage_id',string(sid), ...
    'helper_id',hid,'edge_name',string(ename),'detail',string(detail));
end
