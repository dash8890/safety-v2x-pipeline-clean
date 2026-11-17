function tests = TestSafetyOffloading
tests = functiontests(localfunctions);
end

function setupOnce(tc)
root=fileparts(fileparts(mfilename('fullpath'))); addpath(root);
tc.TestData.root=root;
tc.TestData.cfg=loadConfig(fullfile(root,"config","smoke.json"));
end

function testConfigurationAndFragmentation(tc)
cfg=tc.TestData.cfg;
verifyGreaterThan(tc,cfg.compute.input_bytes,cfg.radio.max_transport_block_bytes);
verifyEqual(tc,ceil(cfg.compute.input_bytes/cfg.radio.max_transport_block_bytes),2);
end

function testDeterministicReplayAndZeroHarm(tc)
cfg=tc.TestData.cfg; trace=generateScenario(cfg,12,41);
a=simulateWorld(cfg,trace,"no_offload"); b=simulateWorld(cfg,trace,"no_offload");
verifyEqual(tc,a.safety,b.safety);
p=pairOutcomes(a.safety,b.safety);
verifyEqual(tc,sum(p.harmful_flip),0);
end

function testPairedKeysAndHarmIdentity(tc)
cfg=tc.TestData.cfg; trace=generateScenario(cfg,12,7);
b=simulateWorld(cfg,trace,"no_offload"); p=simulateWorld(cfg,trace,"unrestricted");
paired=pairOutcomes(b.safety,p.safety);
verifyEqual(tc,height(paired),height(b.safety));
fb=mean(~paired.baseline_success); fp=mean(~paired.policy_success);
h=mean(paired.harmful_flip); g=mean(paired.avoided_failure);
verifyEqual(tc,fp-fb,h-g,'AbsTol',1e-12);
end

function testTaskOwnershipAndDeadlines(tc)
cfg=tc.TestData.cfg; trace=generateScenario(cfg,12,9);
r=simulateWorld(cfg,trace,"unrestricted");
verifyEqual(tc,numel(unique(r.tasks.task_id)),height(r.tasks));
done=r.tasks.completed_on_time;
verifyTrue(tc,all(r.tasks.result_delivered_ms(done)<=r.tasks.deadline_ms(done)));
verifyTrue(tc,all(r.tasks.destination>=0));
end

function testInputAndResultCausality(tc)
cfg=tc.TestData.cfg; trace=generateScenario(cfg,12,12);
r=simulateWorld(cfg,trace,"unrestricted");
remote=r.tasks.action=="offload" & ~isnan(r.tasks.cpu_start_ms);
verifyTrue(tc,all(r.tasks.cpu_start_ms(remote)>=r.tasks.input_delivered_ms(remote)));
hasResult=remote & ~isnan(r.tasks.result_delivered_ms);
verifyTrue(tc,all(r.tasks.result_delivered_ms(hasResult)>=r.tasks.cpu_finish_ms(hasResult)));
end

function testNoHiddenCounterfactualInDecisionLog(tc)
cfg=tc.TestData.cfg; trace=generateScenario(cfg,12,5);
r=simulateWorld(cfg,trace,"risk_heuristic");
names=lower(string(r.decisions.Properties.VariableNames));
verifyFalse(tc,any(contains(names,"baseline_success")|contains(names,"counterfactual")|contains(names,"future")));
end

function testRiskFallback(tc)
cfg=tc.TestData.cfg; cfg.control.risk_ood_cbr=0;
trace=generateScenario(cfg,12,17); r=simulateWorld(cfg,trace,"risk_heuristic");
verifyFalse(tc,any(r.tasks.action=="offload"));
end

function testStableUniform(tc)
u1=stableUniform(2,"decode",10,4); u2=stableUniform(2,"decode",10,4);
verifyEqual(tc,u1,u2); verifyGreaterThanOrEqual(tc,u1,0); verifyLessThan(tc,u1,1);
end

function testIntentionalRadioFailureStopsRemoteExecution(tc)
cfg=tc.TestData.cfg; cfg.radio.tx_power_dbm=-200;
trace=generateScenario(cfg,12,22); r=simulateWorld(cfg,trace,"unrestricted");
remote=r.tasks.action=="offload";
verifyTrue(tc,any(remote));
verifyFalse(tc,any(r.tasks.completed_on_time(remote)));
verifyTrue(tc,all(isnan(r.tasks.cpu_start_ms(remote))));
end

function testExpiryNeverCountsAsCompletion(tc)
cfg=tc.TestData.cfg; cfg.compute.deadline_ms=0.25;
trace=generateScenario(cfg,12,23); r=simulateWorld(cfg,trace,"unrestricted");
verifyFalse(tc,any(r.tasks.completed_on_time));
verifyEqual(tc,height(r.tasks),numel(r.tasks.failure_reason));
end

function testCpuServiceRespectsCycleBudget(tc)
cfg=tc.TestData.cfg; trace=generateScenario(cfg,12,24);
r=simulateWorld(cfg,trace,"unrestricted");
served=~isnan(r.tasks.cpu_finish_ms);
ids=find(served);
for k=1:numel(ids)
    tid=ids(k);
    if r.tasks.action(tid)=="local", owner=r.tasks.origin_id(tid); else, owner=r.tasks.destination(tid); end
    observed=(r.tasks.cpu_finish_ms(tid)-r.tasks.cpu_start_ms(tid))/1000;
    required=r.tasks.cycles(tid)/trace.cpu_rate(owner);
    verifyGreaterThanOrEqual(tc,observed+cfg.radio.slot_duration_ms/1000,required);
end
end

function testApplicationActionsDoNotWriteReservations(tc)
cfg=tc.TestData.cfg; trace=generateScenario(cfg,12,25);
r=simulateWorld(cfg,trace,"edf");
names=lower(string(r.decisions.Properties.VariableNames));
verifyFalse(tc,any(contains(names,"subchannel")|contains(names,"reservation")|contains(names,"grant")));
verifyEqual(tc,sort(unique(r.decisions.scenario_id)),trace.seed);
end

