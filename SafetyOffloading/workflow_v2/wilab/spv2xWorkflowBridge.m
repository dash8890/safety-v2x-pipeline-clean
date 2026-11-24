function varargout=spv2xWorkflowBridge(op,varargin)
%SPV2XWORKFLOWBRIDGE Hybrid WiLab Mode-2 <-> WorkflowEngine (R3).
% Radio: pending_tx -> Mode-2 -> AfterTX/Reception.
% CPU: WorkflowCpuQueue. No free helper oracle.
switch string(op)
    case "attach", varargout{1}=attach(varargin{:});
    case "tick", [varargout{1},varargout{2}]=tick(varargin{:});
    case "reception", varargout{1}=reception(varargin{:});
    case "finalize", varargout{1}=finalize(varargin{:});
    otherwise, error('spv2xWorkflowBridge:UnknownOp','%s',op);
end
end

function s=attach(s,simParams,positionManagement)
% Called from WiLab Init after common overlay fields exist.
assert(startsWith(s.policy,"workflow_"),'not a workflow policy');
cfg=s.cfg;
origin=s.origin_ids(1);
helpers=s.helper_ids(:);
% Prefer geometrically nearest helpers so Mode-2 contact is feasible (R3).
nHelp=2;
if isfield(cfg.workflow,'helper_count'), nHelp=double(cfg.workflow.helper_count); end
if isfield(positionManagement,'distanceReal') && ~isempty(helpers)
    d=arrayfun(@(h)positionManagement.distanceReal(origin,h),helpers);
    d(~isfinite(d))=inf;
    [~,ord]=sort(d);
    helpers=helpers(ord);
end
helpers=helpers(1:min(numel(helpers),nHelp));
assert(numel(helpers)>=1,'need at least one helper vehicle');

map=[origin; helpers(:)]; % engine vehicle index i <-> WiLab ID map(i)
hz=zeros(numel(map),1);
for i=1:numel(map)
    hz(i)=s.cpu_nominal_rate(map(i));
    if hz(i)<=0, hz(i)=5e8; end
end
relDl=0.5;
if isfield(cfg.workflow,'relative_deadline_s'), relDl=double(cfg.workflow.relative_deadline_s); end
% Optional exact CPU vector (engine indices) matching Python causal CFG.
if isfield(cfg.workflow,'cpu_hz')
    ch=double(cfg.workflow.cpu_hz(:));
    for i=1:min(numel(hz),numel(ch)), hz(i)=ch(i); end
end
wcfg=struct('origin_id',1,'n_vehicles',numel(map),'cpu_hz',hz, ...
    'tb_bytes',cfg.radio.max_transport_block_bytes,'rel_deadline_s',relDl, ...
    'ctrl_delay_s',0,'default_link_delay_s',0,'slot_s',0.0005, ...
    'max_admitted_cycles',2*hz(1)*relDl);
if isfield(cfg.workflow,'stage_cycles')
    wcfg.stage_cycles=double(cfg.workflow.stage_cycles(:))';
end
W=WorkflowEngine('new',wcfg);
W.mode="wilab_mode2";
W.wilab_id_map=map;
W.radio_mode="wilab_mode2";
W.seed=simParams.seed;
W.horizon_s=simParams.simulationTime;
if isfield(cfg.workflow,'radio_mode') && string(cfg.workflow.radio_mode)=="abstract_link"
    W.mode="abstract_link";
    W.radio_mode="abstract_link";
    W.default_link_delay_s=getf(cfg.workflow,'abstract_link_delay_s',0.015);
    W.ctrl_delay_s=getf(cfg.workflow,'abstract_ctrl_delay_s',0.005);
end
% Phased value utility (causal DRL scenario)
W.phased_values=isfield(cfg.workflow,'phased_values') && logical(cfg.workflow.phased_values);
W.value_high=getf(cfg.workflow,'value_high',12);
W.value_low=getf(cfg.workflow,'value_low',1);
W.p_high_early=getf(cfg.workflow,'p_high_early',0.05);
W.p_high_late=getf(cfg.workflow,'p_high_late',0.75);
W.phase_split_s=getf(cfg.workflow,'phase_split_s',0.16);
W.advertisement_ttl_s=getf(cfg.workflow,'advertisement_ttl_s',0.30);
W.advertisement_period_s=getf(cfg.workflow,'advertisement_period_s',0.10);
if isfield(cfg.workflow,'fresh_bc_weights_json')
    W.fresh_bc_weights_json=string(cfg.workflow.fresh_bc_weights_json);
elseif string(s.policy)=="workflow_fresh_ddqn"
    W.fresh_bc_weights_json=fullfile(fileparts(mfilename('fullpath')),'..','drl','outputs','fresh_candidate_bc.json');
end
if isfield(cfg.workflow,'ddqn_weights_json')
    wp=string(cfg.workflow.ddqn_weights_json);
    if ~isfile(wp)
        % resolve relative to SafetyOffloading or workflow_v2
        cand={fullfile(fileparts(mfilename('fullpath')),'..',char(wp)), ...
              fullfile(fileparts(mfilename('fullpath')),'..','drl','outputs','ddqn_causal_dueling.weights.json')};
        for ci=1:numel(cand)
            if isfile(cand{ci}), wp=string(cand{ci}); break; end
        end
    end
    W.ddqn_weights_json=wp;
end
if isfield(cfg.workflow,'ddqn_admission_weights_json')
    W.ddqn_admission_weights_json=string(cfg.workflow.ddqn_admission_weights_json);
end
if isfield(cfg.workflow,'ddqn_placement_weights_json')
    W.ddqn_placement_weights_json=string(cfg.workflow.ddqn_placement_weights_json);
end
W.last_contact_change_t=zeros(1,numel(map));
W.last_adv_rx_t=-inf(1,numel(map));
W.next_adv_s=zeros(1,numel(map));
W.adv_seq=zeros(1,numel(map));
W.demo_log=struct('t_s',{},'action_idx',{},'obs',{},'mask',{});
W.collect_demos=isfield(cfg.workflow,'collect_demos') && logical(cfg.workflow.collect_demos);
W.collect_fresh_demos=isfield(cfg.workflow,'collect_fresh_demos') && logical(cfg.workflow.collect_fresh_demos);
W.fresh_demo_log=struct('t_s',{},'candidate_features',{},'action_index',{},'kind',{});
W.ddqn_eps=0;
if isfield(cfg.workflow,'ddqn_eps'), W.ddqn_eps=double(cfg.workflow.ddqn_eps); end
W.ddqn_dagger_beta=0;
if isfield(cfg.workflow,'ddqn_dagger_beta'), W.ddqn_dagger_beta=double(cfg.workflow.ddqn_dagger_beta); end
W.ddqn_teacher_admission=false;
if isfield(cfg.workflow,'ddqn_teacher_admission')
    W.ddqn_teacher_admission=logical(cfg.workflow.ddqn_teacher_admission);
end
W.ddqn_high_active_cap=6;
if isfield(cfg.workflow,'ddqn_high_active_cap')
    W.ddqn_high_active_cap=double(cfg.workflow.ddqn_high_active_cap);
end
W.train_log=struct('obs',{},'action_idx',{},'reward',{},'next_obs',{},'done',{},'mask',{},'next_mask',{});
W.train_pending_idx=0;
W.train_pending_value=0;
W.pending_tr=[];
W.collect_transitions=isfield(cfg.workflow,'collect_transitions') && logical(cfg.workflow.collect_transitions);
W.drop_reward_bonus=0.0;
if isfield(cfg.workflow,'drop_reward_bonus'), W.drop_reward_bonus=double(cfg.workflow.drop_reward_bonus); end
W.high_value_admit_cap=3;
if isfield(cfg.workflow,'high_value_admit_cap'), W.high_value_admit_cap=double(cfg.workflow.high_value_admit_cap); end

% Background load as real CPU jobs (frozen rho); arrivals scheduled then enqueued at tick.
rho=0.30;
if isfield(cfg.workflow,'cpu_offered_load'), rho=double(cfg.workflow.cpu_offered_load); end
bgCycles=12e6;
if isfield(cfg.compute,'background_job_cycles'), bgCycles=double(cfg.compute.background_job_cycles); end
W.bg_plan=struct('vehicle',{},'t_s',{},'cycles',{});
if rho>0
    for i=1:numel(map)
        lam=rho*hz(i)/bgCycles;
        t=0.05; k=0;
        while true
            k=k+1;
            u=stableUniformLocal(simParams.seed,"wf-bg",map(i),k);
            t=t+(-log(max(u,realmin))/max(lam,1e-9));
            if t>=simParams.simulationTime, break; end
            W.bg_plan(end+1)=struct('vehicle',i,'t_s',t,'cycles',bgCycles); %#ok<AGROW>
        end
    end
end
% Optional one-shot origin background matching Python bg_cycles_origin
if isfield(cfg.workflow,'initial_bg_cycles_origin')
    bg0=double(cfg.workflow.initial_bg_cycles_origin);
    if bg0>0
        W.bg_plan(end+1)=struct('vehicle',1,'t_s',0.0,'cycles',bg0); %#ok<AGROW>
    end
end

% Workflow arrivals (frozen)
arr=[];
if isfield(cfg.workflow,'arrival_times_s')
    arr=double(cfg.workflow.arrival_times_s(:));
else
    arr=0.10; % single workflow default for smoke/pilot
end
W.arrival_plan=arr;
W.arrival_cursor=1;
W.policy=s.policy;
W.decision_log=struct('t_s',{},'policy',{},'action',{},'workflow_id',{},'detail',{});
W.sim_end_s=simParams.simulationTime;
W.position_ref=positionManagement;
W.release_mode="controlled";
if isfield(cfg.workflow,'release_mode'), W.release_mode=string(cfg.workflow.release_mode); end
if contains(string(s.policy),"immediate"), W.release_mode="immediate"; end
W.force_scripted_contact=isfield(cfg.workflow,'force_scripted_contact') && logical(cfg.workflow.force_scripted_contact);
W.contact_schedule=struct('t_s',{},'links',{});
if isfield(cfg.workflow,'contact_schedule')
    cs=cfg.workflow.contact_schedule;
    for ci=1:numel(cs)
        L=cs(ci).links;
        if iscell(L)
            L=vertcat(L{:});
        end
        L=double(L);
        if size(L,2)~=3 && numel(L)>=3
            L=reshape(L,3,[])';
        end
        W.contact_schedule(end+1)=struct('t_s',double(cs(ci).t_s),'links',L); %#ok<AGROW>
    end
end
% Seed contact
rangeM=400;
if isfield(cfg.compute,'helper_range_m'), rangeM=double(cfg.compute.helper_range_m); end
if W.force_scripted_contact
    W=applyContactSchedule(W,0);
else
    W=updateContacts(W,positionManagement,rangeM);
end

s.workflow=W;
s.workflow_enabled=true;
% Expand current/empty_meta so workflow metas assign into s.current(v).
wfMeta=WorkflowMode2Adapter('empty_meta');
s.empty_meta=wfMeta;
s.current=repmat(wfMeta,size(s.current));
s.tasks=struct('task_id',{},'origin',{},'release_ms',{},'deadline_ms',{}, ...
    'input_bytes',{},'output_bytes',{},'cycles',{},'value',{},'action',{}, ...
    'helper',{},'status',{},'input_fragments',{},'input_received',{}, ...
    'output_fragments',{},'output_received',{},'cpu_owner',{},'cpu_start_ms',{}, ...
    'cpu_finish_ms',{},'input_delivered_ms',{},'result_delivered_ms',{},'decision_id',{});
s.release_cursor=1;
s.use_joint_scheduler=false;
arrCol=arr(:);
s.next_tick_s=min([0.05; arrCol; 0.01]);
end

function [stationManagement,timeManagement]=tick(stationManagement,timeManagement,positionManagement,simParams,phyParams)
s=stationManagement.spv2x; W=s.workflow; nowS=timeManagement.timeNow; nowMs=nowS*1000;
ids=stationManagement.activeIDsCV2X(:);

% Contact: scripted schedule (R3-M) or geometry
rangeM=400;
if isfield(s.cfg.compute,'helper_range_m'), rangeM=double(s.cfg.compute.helper_range_m); end
if isfield(W,'force_scripted_contact') && W.force_scripted_contact
    W=applyContactSchedule(W,nowS);
else
    W=updateContacts(W,positionManagement,rangeM);
end

% Background arrivals due
keep=true(1,numel(W.bg_plan));
for i=1:numel(W.bg_plan)
    if W.bg_plan(i).t_s<=nowS+1e-12
        W.now_s=W.bg_plan(i).t_s;
        W=WorkflowEngine('enqueue_background',W,W.bg_plan(i).vehicle, ...
            W.bg_plan(i).cycles,W.bg_plan(i).t_s+1.0);
        keep(i)=false;
    end
end
W.bg_plan=W.bg_plan(keep);
W.now_s=nowS;

% Helpers advertise their own queue state over the same Mode-2 application
% queue used by commits and data.  The origin learns it only on successful RX.
advPeriod=0.100;
if isfield(W,'advertisement_period_s'), advPeriod=W.advertisement_period_s; end
for helper=1:numel(W.cpu)
    if helper==W.origin_id || nowS+1e-12<W.next_adv_s(helper), continue; end
    S=W.cpu(helper); q=0; r=0;
    if ~isempty(S.running), r=S.running.remaining; q=q+r; end
    if ~isempty(S.queue)
        for qi=1:numel(S.queue), q=q+S.queue(qi).remaining; end
    end
    W.adv_seq(helper)=W.adv_seq(helper)+1;
    state=struct('nominal_cpu_hz',S.nominal_hz,'available_cpu_hz',S.available_hz, ...
        'queued_cycles',q,'running_remaining_cycles',r,'admitted_pending_cycles',0, ...
        'willingness',W.willingness(helper),'seq',W.adv_seq(helper));
    msg=WorkflowControlProtocol('make_adv',helper,state,nowS*1000,W.advertisement_ttl_s*1000);
    [W.pending_tx,~]=WorkflowControlProtocol('enqueue_ctrl',W.pending_tx,helper,msg,W.tb_bytes);
    W.next_adv_s(helper)=nowS+advPeriod;
end

% Workflow arrivals
while W.arrival_cursor<=numel(W.arrival_plan) && W.arrival_plan(W.arrival_cursor)<=nowS+1e-12
    W.now_s=nowS;
    [W,~]=WorkflowEngine('create_workflow',W,W.arrival_plan(W.arrival_cursor));
    W.arrival_cursor=W.arrival_cursor+1;
end

% Advance engine CPU / abstract radio to now (wilab mode skips scripted radio)
W.now_s=nowS;
if W.mode=="wilab_mode2"
    W=advanceCpuOnly(W,nowS);
else
    W=WorkflowEngine('advance_to',W,nowS);
end

% Controller
W=runController(W,nowS);

% Push engine pending_tx into WiLab only in wilab_mode2 (abstract uses scripted RX).
if W.mode=="wilab_mode2"
    for ei=1:numel(W.wilab_id_map)
        wid=W.wilab_id_map(ei);
        if ei<=numel(W.pending_tx) && ~isempty(W.pending_tx{ei})
            items=W.pending_tx{ei};
            for k=1:numel(items)
                items(k)=toWilabMeta(items(k),W);
            end
            if isempty(s.pending_tx{wid})
                s.pending_tx{wid}=items(:);
            else
                s.pending_tx{wid}=[s.pending_tx{wid}(:); items(:)];
            end
            W.pending_tx{ei}=[];
        end
    end
    for v=ids'
        if isempty(s.pending_tx{v}), continue; end
        if stationManagement.pckBuffer(v)>0 || s.current(v).valid, continue; end
        if timeManagement.timeNextPacket(v)<=nowS+phyParams.TTI+1e-10, continue; end
        item=s.pending_tx{v}(1); s.pending_tx{v}(1)=[];
        stationManagement.pckBuffer(v)=1; stationManagement.pckNextAttempt(v)=1;
        timeManagement.timeLastPacket(v)=nowS;
        if isfield(stationManagement,'newDataIndicator'), stationManagement.newDataIndicator(v)=true; end
        s.current(v)=item;
    end
else
    % abstract_link: drain scripted events already applied in advance_to; clear app TX
    for ei=1:numel(W.pending_tx), W.pending_tx{ei}=[]; end
end

% Next wake: never schedule in the past (WiLab "time goes back" assert).
nextCpu=inf;
for i=1:numel(W.cpu)
    if ~isempty(W.cpu(i).running) && W.cpu(i).available_hz>0
        nextCpu=min(nextCpu,W.cpu(i).last_accounted_s+W.cpu(i).running.remaining/W.cpu(i).available_hz);
    end
end
nextArr=inf;
if W.arrival_cursor<=numel(W.arrival_plan), nextArr=W.arrival_plan(W.arrival_cursor); end
nextBg=inf;
if ~isempty(W.bg_plan), nextBg=min([W.bg_plan.t_s]); end
nextEv=inf;
if ~isempty(W.event_q)
    nextEv=min(cellfun(@(e)e.t_s,W.event_q));
end
ctrlTick=nowS+0.010;
cand=[nextCpu,nextArr,nextBg,nextEv,ctrlTick,nowS+0.050];
cand=cand(isfinite(cand) & cand>nowS+1e-12);
if isempty(cand), s.next_tick_s=nowS+0.010; else, s.next_tick_s=min(cand); end
s.workflow=W;
stationManagement.spv2x=s;
end

function stationManagement=reception(stationManagement,timeManagement,txIDs,fate)
s=stationManagement.spv2x; W=s.workflow; nowS=timeManagement.timeNow; nowMs=nowS*1000;
W.now_s=nowS;
for tx=txIDs(:)'
    meta=s.current(tx); if ~meta.valid, continue; end
    rows=fate(fate(:,1)==tx,:);
    for i=1:size(rows,1)
        rx=rows(i,2); success=logical(rows(i,5));
        if ~success, continue; end
        if ~(meta.kind=="workflow_data" || meta.kind=="workflow_ctrl"), continue; end
        % Map WiLab IDs to engine indices
        if meta.kind=="workflow_ctrl"
            msg=ctrlFromMeta(meta);
            srcE=wilabToEngine(W,meta.src);
            dstE=wilabToEngine(W,rx);
            % Remap embedded vehicle ids in protocol fields
            if msg.helper_id>0, msg.helper_id=max(1,wilabToEngine(W,msg.helper_id)); end
            if msg.origin_id>0, msg.origin_id=max(1,wilabToEngine(W,msg.origin_id)); end
            if dstE>0
                W=WorkflowEngine('inject_ctrl_rx',W,msg,srcE,dstE);
            end
            W.log=WorkflowMetrics('log',W.log,nowS,"FRAGMENT_RX",'vehicle_id',dstE, ...
                'detail',"ctrl "+string(meta.ctrl_type));
        else
            if rx~=meta.dst, continue; end
            edgeName=edgeNameFromId(W,meta.workflow_id,meta.edge_id);
            if strlength(edgeName)==0, continue; end
            W=WorkflowEngine('inject_fragment_rx',W,meta.workflow_id,edgeName, ...
                meta.fragment,meta.fragments,meta.message_id);
        end
        if s.log_radio_events
            s.radio_rows(end+1,:)={s.scenario_id,nowMs,tx,meta.packet_id,char(meta.kind), ...
                meta.workflow_id,char(string(meta.edge_id)),meta.fragment,rx,rows(i,4),rows(i,3),true,char(s.world)}; %#ok<AGROW>
        end
    end
    if stationManagement.pckTxOccurring(tx)>=stationManagement.cv2xNumberOfReplicas(tx)
        s.current(tx)=s.empty_meta;
        if ~isempty(s.meta_queue{tx})
            s.current(tx)=s.meta_queue{tx}(1); s.meta_queue{tx}(1)=[];
        end
    end
end
s.workflow=W; stationManagement.spv2x=s;
end

function stationManagement=finalize(stationManagement,outParams,simParams)
s=stationManagement.spv2x; W=s.workflow;
folder=fullfile(outParams.outputFolder,"workflow_v2");
if ~isfolder(folder), mkdir(folder); end
summ=WorkflowEngine('summary',W);
% Event log
n=numel(W.log);
rows=cell(n,7);
for i=1:n
    e=W.log(i);
    rows(i,:)={e.t_s,char(e.type),e.workflow_id,char(string(e.stage_id)),e.edge_id,e.vehicle_id,char(string(e.detail))};
end
ev=cell2table(rows,'VariableNames',{'t_s','type','workflow_id','stage_id','edge_id','vehicle_id','detail'});
writetable(ev,fullfile(folder,"events.csv"));
% Workflows (+ value for utility accounting)
nw=numel(W.workflows); wrows=cell(nw,6);
for i=1:nw
    w=W.workflows(i);
    wrows(i,:)={w.workflow_id,char(w.status),w.release_s,w.absolute_deadline_s,w.completion_s,w.value};
end
writetable(cell2table(wrows,'VariableNames',{'workflow_id','status','release_s','deadline_s','completion_s','value'}), ...
    fullfile(folder,"workflows.csv"));
% Decisions
nd=numel(W.decision_log); drows=cell(nd,5);
for i=1:nd
    d=W.decision_log(i);
    drows(i,:)={d.t_s,char(string(d.policy)),char(string(d.action)),d.workflow_id,char(string(d.detail))};
end
writetable(cell2table(drows,'VariableNames',{'t_s','policy','action','workflow_id','detail'}), ...
    fullfile(folder,"decisions.csv"));
% Summary JSON
nSucc=0; nExp=0; nRemote=0; totalValue=0;
for i=1:nw
    if W.workflows(i).status=="SUCCEEDED"
        nSucc=nSucc+1;
        totalValue=totalValue+double(W.workflows(i).value);
    end
    if W.workflows(i).status=="EXPIRED", nExp=nExp+1; end
end
for i=1:numel(W.stages)
    if W.stages(i).placement_id>0 && W.stages(i).placement_id~=W.origin_id && ...
            W.stages(i).status=="DONE"
        nRemote=nRemote+1;
    end
end
fid=fopen(fullfile(folder,"summary.json"),'w');
fprintf(fid,'%s',jsonencode(struct('policy',char(s.policy),'seed',simParams.seed, ...
    'n_workflows',nw,'n_success',nSucc,'n_expired',nExp,'n_remote_stages_done',nRemote, ...
    'total_value',totalValue,'cpu_served',summ.cpu_served, ...
    'radio_data_fragments',summ.radio_data_fragments, ...
    'radio_mode',char(W.radio_mode)),'PrettyPrint',true));
fclose(fid);
% Fresh candidate-index BC examples. Each row is one executable candidate
% selected by the causal teacher, so no legacy action IDs enter P3 training.
if isfield(W,'fresh_demo_log') && ~isempty(W.fresh_demo_log)
    % Keep each variable-size candidate matrix intact.  Flattening to only
    % the chosen row would make candidate-index BC impossible to train.
    fid=fopen(fullfile(folder,"fresh_demos.json"),'w'); fprintf(fid,'%s',jsonencode(W.fresh_demo_log)); fclose(fid);
end
% WiLab BC demos (obs/action/mask)
if isfield(W,'demo_log') && ~isempty(W.demo_log)
    nd=numel(W.demo_log);
    nObs=numel(W.demo_log(1).obs);
    obsM=zeros(nd,nObs); maskM=zeros(nd,5); actV=zeros(nd,1); tV=zeros(nd,1);
    for i=1:nd
        tV(i)=W.demo_log(i).t_s;
        actV(i)=W.demo_log(i).action_idx;
        obsM(i,:)=W.demo_log(i).obs;
        maskM(i,:)=W.demo_log(i).mask;
    end
    dem=struct('t_s',tV,'action_idx',actV,'obs',obsM,'mask',maskM);
    fid=fopen(fullfile(folder,"demos.json"),'w');
    fprintf(fid,'%s',jsonencode(dem)); fclose(fid);
end
% Online DDQN transitions from this WiLab episode
if isfield(W,'train_log') && ~isempty(W.train_log)
    nt=numel(W.train_log);
    % Finalize a decision for which no later decision state was reached.
    if isfield(W,'train_pending_idx') && W.train_pending_idx>0
        pi=W.train_pending_idx;
        [termObs,termMask,~]=WorkflowCausalObs(W);
        W.train_log(pi).reward=totalUtility(W)-W.train_pending_value;
        W.train_log(pi).next_obs=termObs(:)';
        W.train_log(pi).next_mask=double(termMask(:))';
    end
    % Mark the episode boundary and retain a small terminal shaping signal.
    W.train_log(nt).done=1;
    W.train_log(nt).reward=W.train_log(nt).reward+0.05*totalValue;
    nObs=numel(W.train_log(1).obs);
    obsM=zeros(nt,nObs); nobsM=zeros(nt,nObs); maskM=zeros(nt,5); nmaskM=zeros(nt,5);
    actV=zeros(nt,1); rV=zeros(nt,1); dV=zeros(nt,1);
    for i=1:nt
        obsM(i,:)=W.train_log(i).obs; nobsM(i,:)=W.train_log(i).next_obs;
        maskM(i,:)=W.train_log(i).mask; nmaskM(i,:)=W.train_log(i).next_mask;
        actV(i)=W.train_log(i).action_idx; rV(i)=W.train_log(i).reward; dV(i)=W.train_log(i).done;
    end
    tr=struct('obs',obsM,'action_idx',actV,'reward',rV,'next_obs',nobsM, ...
        'done',dV,'mask',maskM,'next_mask',nmaskM,'total_value',totalValue);
    fid=fopen(fullfile(folder,"transitions.json"),'w');
    fprintf(fid,'%s',jsonencode(tr)); fclose(fid);
end
% Also write empty safety_offloading tasks for legacy finalize compatibility
legacyFolder=fullfile(outParams.outputFolder,"safety_offloading");
if ~isfolder(legacyFolder), mkdir(legacyFolder); end
if ~isfile(fullfile(legacyFolder,"tasks.csv"))
    writetable(table(),fullfile(legacyFolder,"tasks.csv"));
end
s.workflow=W; stationManagement.spv2x=s;
end

%% ---- helpers ----
function W=updateContacts(W,pos,rangeM)
if ~isfield(pos,'distanceReal'), return; end
for a=1:numel(W.wilab_id_map)
    for b=a+1:numel(W.wilab_id_map)
        ia=W.wilab_id_map(a); ib=W.wilab_id_map(b);
        d=pos.distanceReal(ia,ib);
        tf=isfinite(d) && d<=rangeM;
        if W.contact(a,b)~=tf
            W=WorkflowEngine('set_contact',W,a,b,tf);
            W=noteContactChange(W,a,b,tf);
        end
    end
end
end

function W=advanceCpuOnly(W,t)
for v=1:numel(W.cpu)
    [W.cpu(v),evs]=WorkflowCpuQueue('settle',W.cpu(v),t);
    for k=1:numel(evs)
        W.now_s=evs{k}.end_s;
        W=WorkflowEngine('__on_cpu_complete',W,evs{k},v);
    end
end
W.now_s=t;
for i=1:numel(W.workflows)
    W=WorkflowEngine('__check_success',W,W.workflows(i).workflow_id);
end
W=WorkflowEngine('__expire',W,t);
end

function W=applyContactSchedule(W,nowS)
% Apply latest schedule entry with t_s <= nowS (engine vehicle indices).
if ~isfield(W,'contact_schedule') || isempty(W.contact_schedule), return; end
idx=0;
for i=1:numel(W.contact_schedule)
    if W.contact_schedule(i).t_s<=nowS+1e-12, idx=i; end
end
if idx==0, return; end
L=W.contact_schedule(idx).links;
for r=1:size(L,1)
    a=L(r,1); b=L(r,2); tf=logical(L(r,3));
    if a>=1 && b>=1 && a<=size(W.contact,1) && b<=size(W.contact,2)
        if W.contact(a,b)~=tf
            W=WorkflowEngine('set_contact',W,a,b,tf);
            W=noteContactChange(W,a,b,tf);
        end
    end
end
end

function W=noteContactChange(W,a,b,tf)
% Causal obs: contact age + advertisement RX proxy when link comes up.
if ~isfield(W,'last_contact_change_t')
    W.last_contact_change_t=zeros(1,size(W.contact,1));
end
if ~isfield(W,'last_adv_rx_t')
    W.last_adv_rx_t=-inf(1,size(W.contact,1));
end
t=W.now_s;
if a==W.origin_id
    W.last_contact_change_t(b)=t;
    if tf, W.last_adv_rx_t(b)=t; end
elseif b==W.origin_id
    W.last_contact_change_t(a)=t;
    if tf, W.last_adv_rx_t(a)=t; end
end
end

function W=runController(W,nowS)
W.now_s=nowS;
pol=string(W.policy);
if pol=="workflow_fresh_ddqn"
    [a,~]=WorkflowPolicyFreshDDQN(W);
    W=WorkflowActionValidator('apply',W,a);
    W.decision_log(end+1)=struct('t_s',nowS,'policy',pol,'action',string(a.kind), ...
        'workflow_id',a.workflow_id,'detail',string(a.detail)); %#ok<AGROW>
    return
end
seqPol=pol=="workflow_value_greedy" || pol=="workflow_drop_low" || ...
    pol=="workflow_drop_low_cap" || pol=="workflow_drop_low_congest" || ...
    pol=="workflow_causal_ddqn";
if seqPol
    W=runSequentialUtilityPolicy(W,nowS,pol);
    return
end
switch pol
    case "workflow_b0_local", acts=WorkflowPolicyLocalEDF(W);
    case "workflow_b1_greedy", acts=WorkflowPolicyStageGreedy(W);
    case "workflow_b2_queue", acts=WorkflowPolicyQueueAware(W);
    case "workflow_b3_mobility", acts=WorkflowPolicyMobilityAware(W);
    case "workflow_b4_joint", acts=WorkflowPolicyJointPlanner(W);
    case "workflow_b4_immediate", acts=WorkflowPolicyJointImmediate(W);
    case "workflow_b5_privileged", acts=WorkflowPolicyPrivileged(W);
    otherwise, acts=WorkflowPolicyLocalEDF(W);
end
W=applyActs(W,nowS,pol,acts);
end

function W=runSequentialUtilityPolicy(W,nowS,pol)
% One admission/placement decision per loop so n_active / obs stay causal.
for k=1:128
    % Finalize the previous decision only when the simulator reaches the next
    % decision state.  The former implementation finalized it immediately,
    % before time/CPU/radio advanced, which discarded all delayed utility.
    if pol=="workflow_causal_ddqn" && isfield(W,'collect_transitions') && W.collect_transitions ...
            && isfield(W,'train_pending_idx') && W.train_pending_idx>0
        [curObs,curMask,curSt]=WorkflowCausalObs(W);
        % An idle bridge tick is not an MDP transition.  Keep the action
        % pending so all utility accumulated before the next real decision is
        % credited to it (or finalize it at the terminal state).
        if ~isempty(curSt)
            pi=W.train_pending_idx;
            W.train_log(pi).reward=totalUtility(W)-W.train_pending_value;
            W.train_log(pi).next_obs=curObs(:)';
            W.train_log(pi).next_mask=double(curMask(:))';
            W.train_pending_idx=0;
        end
    end
    if (pol=="workflow_drop_low" || (isfield(W,'collect_demos') && W.collect_demos))
        W=recordDemoIfDeciding(W,nowS,pol);
    end
    preValue=totalUtility(W);
    switch pol
        case "workflow_value_greedy", acts=WorkflowPolicyValueGreedy(W);
        case "workflow_drop_low", acts=WorkflowPolicyDropLow(W);
        case "workflow_drop_low_cap", acts=WorkflowPolicyDropLowCap(W);
        case "workflow_drop_low_congest", acts=WorkflowPolicyDropLowCongest(W);
        case "workflow_causal_ddqn", acts=WorkflowPolicyCausalDDQN(W);
        otherwise, acts=WorkflowPolicyValueGreedy(W);
    end
    if isempty(acts), break; end
    onlyRelease=true; sawStop=false; aIdx=0;
    for i=1:numel(acts)
        if string(acts(i).type)~="RELEASE_EDGE", onlyRelease=false; end
        if string(acts(i).type)=="STOP", sawStop=true; end
        if aIdx==0 && string(acts(i).type)~="RELEASE_EDGE"
            aIdx=actionToIdx(acts(i));
        end
    end
    % capture obs/mask from DDQN decision stash
    obs=[]; mask=[];
    if pol=="workflow_causal_ddqn" && isfield(W,'collect_transitions') && W.collect_transitions
        global DDQN_LAST_DECISION %#ok<TLEV>
        if ~isempty(DDQN_LAST_DECISION)
            obs=DDQN_LAST_DECISION.obs; mask=DDQN_LAST_DECISION.mask; aIdx=DDQN_LAST_DECISION.action_idx;
        end
    end
    W=applyActs(W,nowS,pol,acts);
    if pol=="workflow_causal_ddqn" && isfield(W,'collect_transitions') && W.collect_transitions && ~isempty(obs)
        [nobs,nmask,~]=WorkflowCausalObs(W);
        r=0.0; % finalized at the next decision after simulated time advances
        % Optional DROP shaping (disable when training to beat DropLow)
        dropBonus=0.0;
        if isfield(W,'drop_reward_bonus'), dropBonus=double(W.drop_reward_bonus); end
        if aIdx==5 && dropBonus~=0
            r=r+dropBonus;
        end
        done=false;
        W.train_log(end+1)=struct('obs',obs,'action_idx',aIdx,'reward',r, ...
            'next_obs',nobs(:)','done',double(done),'mask',mask,'next_mask',double(nmask(:))'); %#ok<AGROW>
        W.train_pending_idx=numel(W.train_log);
        W.train_pending_value=preValue;
    end
    if onlyRelease || sawStop, break; end
end
end

function v=totalUtility(W)
v=0;
for i=1:numel(W.workflows)
    if W.workflows(i).status=="SUCCEEDED"
        v=v+double(W.workflows(i).value);
    end
end
end

function W=recordDemoIfDeciding(W,nowS,pol)
% Log causal obs + teacher action for WiLab BC.
% Teachers: Drop-Low (PDP) or Drop-Low-Congest (Outcome-A admission rule).
if pol~="workflow_drop_low" && pol~="workflow_drop_low_congest" && pol~="workflow_causal_ddqn"
    return
end
[obs,mask,st]=WorkflowCausalObs(W);
if isempty(st), return; end
if pol=="workflow_drop_low_congest" || pol=="workflow_causal_ddqn"
    acts=WorkflowPolicyDropLowCongest(W);
else
    acts=WorkflowPolicyDropLow(W);
end
if isempty(acts), return; end
a=acts(1);
if string(a.type)=="RELEASE_EDGE", return; end
idx=actionToIdx(a);
if idx<1, return; end
if ~isfield(W,'demo_log'), W.demo_log=struct('t_s',{},'action_idx',{},'obs',{},'mask',{}); end
W.demo_log(end+1)=struct('t_s',nowS,'action_idx',idx,'obs',obs(:)','mask',double(mask(:))'); %#ok<AGROW>
end

function idx=actionToIdx(a)
switch string(a.type)
    case "COMMIT_LOCAL", idx=1;
    case "REQUEST_REMOTE"
        if a.helper_id==2, idx=2; elseif a.helper_id==3, idx=3; else, idx=1; end
    case "STOP", idx=4;
    case "DROP", idx=5;
    otherwise, idx=0;
end
end

function W=applyActs(W,nowS,pol,acts)
for i=1:numel(acts)
    % Legacy policies may still propose their old action structs, but the
    % bridge executes only the fresh union.  In particular, an already
    % committed stage is not re-committed on every WiLab callback.
    a=WorkflowActionValidator('legacy_to_fresh',W,acts(i));
    if ~WorkflowActionValidator('validate',W,a)
        continue
    end
    if isfield(W,'collect_fresh_demos') && W.collect_fresh_demos
        c=WorkflowCandidateBuilder(W); X=WorkflowFreshCandidateFeatures(W,c); ix=0;
        for ci=1:numel(c)
            if c(ci).kind==a.kind && c(ci).workflow_id==a.workflow_id && ...
                    c(ci).stage_id==a.stage_id && c(ci).helper_id==a.helper_id && c(ci).edge_id==a.edge_id
                ix=ci; break
            end
        end
        if ix>0
            W.fresh_demo_log(end+1)=struct('t_s',nowS,'candidate_features',X, ...
                'action_index',ix,'kind',a.kind); %#ok<AGROW>
        end
    end
    W=WorkflowActionValidator('apply',W,a);
    W.decision_log(end+1)=struct('t_s',nowS,'policy',pol,'action',string(a.kind), ...
        'workflow_id',a.workflow_id,'detail',string(a.detail)); %#ok<AGROW>
end
end

function m=toWilabMeta(m,W)
% Ensure fields legacy AfterTX/Reception expect.
if ~isfield(m,'task_id'), m.task_id=0; end
if ~isfield(m,'stage'), m.stage=""; end
if ~isfield(m,'packet_id') || m.packet_id==0
    m.packet_id=1e6+double(m.message_id)*10+double(m.fragment);
end
% Remap engine indices -> WiLab IDs
if m.src>=1 && m.src<=numel(W.wilab_id_map), m.src=W.wilab_id_map(m.src); end
if m.dst>=1 && m.dst<=numel(W.wilab_id_map), m.dst=W.wilab_id_map(m.dst); end
if isfield(m,'helper_id') && m.helper_id>=1 && m.helper_id<=numel(W.wilab_id_map)
    m.helper_id=W.wilab_id_map(m.helper_id);
end
if isfield(m,'origin_id') && m.origin_id>=1 && m.origin_id<=numel(W.wilab_id_map)
    m.origin_id=W.wilab_id_map(m.origin_id);
end
end

function msg=ctrlFromMeta(meta)
% Reconstruct protocol message; remap WiLab vehicle IDs → engine indices at inject site.
msg=struct('ctrl_type',string(meta.ctrl_type),'helper_id',meta.helper_id,'origin_id',meta.origin_id, ...
    'workflow_id',meta.workflow_id,'stage_id',string(meta.stage_id),'cycles',meta.cycles, ...
    'deadline_ms',meta.deadline_ms,'input_bytes',meta.input_bytes,'nonce',meta.nonce, ...
    'generated_ms',meta.created_ms,'payload_bytes',meta.bytes,'commitment_id',meta.commitment_id, ...
    'willingness',logical(meta.willingness),'seq',meta.seq,'nominal_cpu_hz',meta.nominal_cpu_hz, ...
    'available_cpu_hz',meta.available_cpu_hz,'queued_cycles',meta.queued_cycles, ...
    'running_remaining_cycles',meta.running_remaining_cycles,'admitted_pending_cycles',meta.admitted_pending_cycles,'ttl_ms',meta.ttl_ms, ...
    'expire_ms',meta.expire_ms,'reason',string(meta.reason));
end

function ei=wilabToEngine(W,wid)
ei=0;
for i=1:numel(W.wilab_id_map)
    if W.wilab_id_map(i)==wid, ei=i; return; end
end
end

function name=edgeNameFromId(W,wfId,edgeId)
name="";
for i=1:numel(W.edges)
    if W.edges(i).workflow_id==wfId && W.edges(i).edge_id==edgeId
        name=W.edges(i).name; return
    end
end
end

function v=getf(s,f,d)
if isstruct(s) && isfield(s,f), v=s.(f); else, v=d; end
end

function u=stableUniformLocal(seed,tag,a,b)
% Lightweight deterministic U(0,1) without depending on legacy path visibility.
persistent cache
if isempty(cache), cache=containers.Map(); end
key=sprintf('%d|%s|%d|%d',seed,tag,a,b);
if isKey(cache,key), u=cache(key); return; end
h=double(mod(uint32(seed)*2654435761+uint32(a)*40503+uint32(b)*2246822519,2^31-1));
u=h/(2^31-1);
if u<=0, u=eps; end
cache(key)=u;
end
