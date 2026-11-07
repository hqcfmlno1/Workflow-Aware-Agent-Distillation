"""User-selected task budgets, independent of the number of model requests."""
from __future__ import annotations

import json
from pathlib import Path

from minisweagent.exceptions import LimitsExceeded
from .step_resume import ResumableAgent
from .campaign import atomic_json, run_task, TaskImageUnavailable

class ToolBudgetAgent(ResumableAgent):
    def __init__(self,*args,tool_call_limit=50,**kwargs):
        if type(tool_call_limit) is not int or tool_call_limit<1:
            raise ValueError('tool_call_limit must be positive')
        self.tool_call_limit=tool_call_limit
        self.tool_calls_started=0
        super().__init__(*args,**kwargs)
        if self.resume_state:
            if 'tool_calls_started' not in self.resume_state:
                raise ValueError('Resume lacks actual tool count; use separate bounded run')
            self.tool_calls_started=self.resume_state['tool_calls_started']
        if self.config.step_limit:
            raise ValueError('Model-call step_limit must be disabled; actual tools are counted')

    def checkpoint(self,phase,*,result=None):
        # Persist the counter in the same atomic write as the action journal.
        self.checkpoint_metadata['tool_calls_started']=self.tool_calls_started
        return super().checkpoint(phase,result=result)

    def check_tool_budget(self):
        if self.tool_calls_started>=self.tool_call_limit:
            raise LimitsExceeded({'role':'exit','content':'ToolCallLimit','extra':{
                'exit_status':'ToolCallLimit','submission':'',
                'tool_calls_started':self.tool_calls_started,'tool_call_limit':self.tool_call_limit}})

    def query(self):
        self.check_tool_budget()
        return super().query()

    def execute_actions(self,message):
        actions=message.get('extra',{}).get('actions',[])
        for index in range(self._next_action,len(actions)):
            if self.tool_calls_started>=self.tool_call_limit:
                self.add_messages(*self.model.format_observation_messages(
                    message,self._outputs,self.get_template_vars()))
                self.check_tool_budget()
            self._next_action=index
            self.tool_calls_started+=1
            self.checkpoint('action_running')
            output=self.env.execute(actions[index])
            self._outputs.append(output)
            self._next_action=index+1
            self.checkpoint('actions')
        added=self.add_messages(*self.model.format_observation_messages(
            message,self._outputs,self.get_template_vars()))
        self._next_action=0; self._outputs=[]
        self.checkpoint('query')
        return added


def run_task_continuing(store,row,resources,infer,grade):
    """Task-local failures are terminal; Ctrl-C keeps step-resume semantics."""
    iid=row['instance_id']
    try:
        run_task(store,row,resources,infer,grade)
    except TaskImageUnavailable:
        store.event('image_unavailable_campaign_paused',instance_id=iid)
        raise
    except Exception as exc:
        state=store.task(iid)
        timeout=isinstance(exc,TimeoutError)
        kind='turn_timeout' if timeout else 'infrastructure_error'
        result={'resolved':False if timeout else None,'kind':kind,
                'error_type':type(exc).__name__,'error':str(exc),
                'official_graded':False}
        folder=Path(state.get('attempt_dir',store.output/'tasks'/iid))
        path=folder/'resume.json'
        if path.exists():
            saved=json.loads(path.read_text())
            saved.update(phase='complete',result=result)
            atomic_json(path,saved)
        atomic_json(folder/'terminal-error.json',result)
        store.put(iid,phase='done',result=result,error=str(exc))
        store.predictions()
        # run_task parks the workspace on exception; terminal tasks release it.
        resources.cleanup()
        store.event('task_error_continue',instance_id=iid,**result)


def paired_summary(store,ids):
    outcomes={iid:store.task(iid) for iid in ids if store.task(iid)['phase']=='done'}
    graded={iid:s['result'] for iid,s in outcomes.items() if s['result'].get('kind')=='graded'}
    infrastructure=sum(s['result'].get('kind')=='infrastructure_error' for s in outcomes.values())
    resolved=sum(bool(s['result'].get('resolved')) for s in outcomes.values())
    return {'total':len(ids),'done':len(outcomes),'pending':len(ids)-len(outcomes),
        'resolved':resolved,'official_graded':len(graded),'infrastructure_errors':infrastructure,
        'resolved_fraction_of_all':resolved/len(ids) if len(ids) else None,
        'complete':len(outcomes)==len(ids),
        'results':{iid:s['result'] for iid,s in outcomes.items()}}
