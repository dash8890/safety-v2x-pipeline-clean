function acts=WorkflowPolicyLocalEDF(W)
%B0: All stages local; EDF via WorkflowCpuQueue; no V2V transfers.
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
for i=1:numel(W.stages)
    st=W.stages(i);
    if st.status=="ELIGIBLE"
        acts(end+1)=act("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,"","local"); %#ok<AGROW>
    end
end
end

function a=act(type,wf,sid,hid,ename,detail)
a=struct('type',string(type),'workflow_id',wf,'stage_id',string(sid), ...
    'helper_id',hid,'edge_name',string(ename),'detail',string(detail));
end
