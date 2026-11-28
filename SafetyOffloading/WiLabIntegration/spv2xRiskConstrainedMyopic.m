function action=spv2xRiskConstrainedMyopic(obs,s,tentative,planQueue)
%SPV2XRISKCONSTRAINEDMYOPIC Emit next action from a P6 myopic plan queue.
% planQueue is a cell/struct array consumed by the epoch loop.  If empty,
% a new joint plan is computed from the deployable observation.
if nargin<4, planQueue=[]; end
action=struct('type',"STOP",'task_id',0,'processor_id',0,'plan',[]);
if ~isempty(planQueue)
    action=planQueue(1);
    action.plan=planQueue(2:end);
    return
end
plan=spv2xRiskConstrainedMyopicPlan(obs,s,tentative);
if isempty(plan)
    action.plan=[];
    return
end
action=plan(1);
action.plan=plan(2:end);
end
