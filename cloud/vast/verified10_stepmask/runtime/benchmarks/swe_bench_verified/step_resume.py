"""Durable mini-SWE-agent step boundaries without replaying completed shell actions."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, InterruptAgentFlow


class UncertainToolExecution(RuntimeError):
    """A command was started but its result was not committed; never blindly replay it."""


class ResumableAgent(DefaultAgent):
    def __init__(self, *args, checkpoint_path, checkpoint_metadata, write_checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_metadata = deepcopy(checkpoint_metadata)
        self.write_checkpoint = write_checkpoint
        self.resume_state = (
            json.loads(self.checkpoint_path.read_text()) if self.checkpoint_path.exists() else None
        )
        if self.resume_state:
            if self.resume_state.get('schema') != 1:
                raise ValueError('Unsupported step checkpoint schema')
            for key in ('instance_id', 'task_sha256', 'run_id', 'image'):
                expected = checkpoint_metadata.get(key)
                if expected is not None and self.resume_state.get(key) != expected:
                    raise ValueError('Step resume identity changed: '+key)
        self._next_action = 0
        self._outputs = []
        self._phase = 'query'

    def checkpoint(self, phase, *, result=None):
        self._phase = phase
        value = {
            **self.checkpoint_metadata,
            'schema': 1,
            'phase': phase,
            'messages': self.messages,
            'n_calls': self.n_calls,
            'cost': self.cost,
            'n_consecutive_format_errors': self.n_consecutive_format_errors,
            'next_action': self._next_action,
            'outputs': self._outputs,
        }
        if result is not None:
            value['result'] = result
        self.write_checkpoint(self.checkpoint_path, value)

    def mark_finished(self, result):
        self.checkpoint('complete', result=result)

    def query(self):
        # Commit the history/counter before incrementing for an attempted call.
        # A failed model response is retried at this same call number.
        self.checkpoint('query')
        message = super().query()
        self._next_action = 0
        self._outputs = []
        self.checkpoint('actions')
        return message

    def execute_actions(self, message):
        actions = message.get('extra', {}).get('actions', [])
        for index in range(self._next_action, len(actions)):
            self._next_action = index
            self.checkpoint('action_running')
            output = self.env.execute(actions[index])
            self._outputs.append(output)
            self._next_action = index + 1
            self.checkpoint('actions')
        observations = self.model.format_observation_messages(
            message, self._outputs, self.get_template_vars()
        )
        added = self.add_messages(*observations)
        self._next_action = 0
        self._outputs = []
        self.checkpoint('query')
        return added

    def run(self, task='', **kwargs):
        self.extra_template_vars |= {'task': task, **kwargs}
        saved = self.resume_state
        if saved:
            if saved['phase'] == 'action_running':
                raise UncertainToolExecution(
                    'Interrupted during shell action '+str(saved.get('next_action', 0))+
                    '; inspect retained container before deciding whether it may be replayed'
                )
            self.messages = deepcopy(saved['messages'])
            self.n_calls = saved['n_calls']
            self.cost = saved.get('cost', 0.0)
            self.n_consecutive_format_errors = saved.get('n_consecutive_format_errors', 0)
            self._next_action = saved.get('next_action', 0)
            self._outputs = deepcopy(saved.get('outputs', []))
            self._phase = saved['phase']
            if self._phase == 'complete':
                return saved.get('result', self.messages[-1].get('extra', {}))
        else:
            self.messages = []
            self.add_messages(
                self.model.format_message(
                    role='system', content=self._render_template(self.config.system_template)
                ),
                self.model.format_message(
                    role='user', content=self._render_template(self.config.instance_template)
                ),
            )
            self.checkpoint('query')
        # Preserve mini-swe-agent 2.4.6's native error handling. Only the starting
        # state and tool journal differ from DefaultAgent.run.
        while True:
            try:
                if self._phase == 'actions':
                    self.execute_actions(self.messages[-1])
                else:
                    self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as exc:
                self.cost += exc.messages[0].get('extra', {}).get('cost', 0.0)
                self.n_consecutive_format_errors += 1
                if (0 < self.config.max_consecutive_format_errors
                        <= self.n_consecutive_format_errors):
                    self.add_messages(*exc.messages, {
                        'role':'exit', 'content':'RepeatedFormatError',
                        'extra':{'exit_status':'RepeatedFormatError','submission':''},
                    })
                else:
                    self.add_messages(*exc.messages)
                self.checkpoint('query')
            except InterruptAgentFlow as exc:
                self.add_messages(*exc.messages)
            except Exception as exc:
                self.handle_uncaught_exception(exc)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get('role') == 'exit':
                result = self.messages[-1].get('extra', {})
                self.mark_finished(result)
                return result


def latest_checkpoint(task_directory):
    paths = sorted(Path(task_directory).glob('attempt-*/resume.json'), reverse=True)
    return json.loads(paths[0].read_text()) if paths else None


def _owned_container_state(saved, resources, *, missing_ok=False):
    container = saved['container_id']
    labels = resources.command('inspect','--format','{{json .Config.Labels}}',container)
    if labels.returncode:
        if missing_ok and 'No such' in labels.stderr:
            return None
        raise RuntimeError('Saved resume container is unavailable; no shell commands were replayed')
    labels = json.loads(labels.stdout or '{}')
    if (labels.get('workflow.bench.run') != resources.run_id
            or labels.get('workflow.bench.task') != saved['instance_id']):
        raise RuntimeError('Refusing to attach to or pause a container with different ownership')
    state = resources.command('inspect','--format','{{json .State}}',container)
    if state.returncode:
        raise RuntimeError('Cannot inspect resume container state')
    return json.loads(state.stdout)


def park_container(saved, resources):
    state = _owned_container_state(saved, resources, missing_ok=True)
    if state is None:
        return False
    if state.get('Running') and not state.get('Paused'):
        result = resources.command('pause',saved['container_id'])
        if result.returncode:
            raise RuntimeError('Cannot pause resume container: '+result.stderr)
    return True


def attach_container(saved, resources):
    import logging
    from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
    state = _owned_container_state(saved, resources)
    container = saved['container_id']
    if state.get('Paused'):
        result = resources.command('unpause',container)
    elif not state.get('Running'):
        result = resources.command('start',container)
    else:
        result = None
    if result is not None and result.returncode:
        raise RuntimeError('Cannot reactivate resume container: '+result.stderr)
    # Attach to the existing filesystem; do not call __init__/_start_container.
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.logger = logging.getLogger('minisweagent.environment')
    env.config = DockerEnvironmentConfig(**saved['environment_config'])
    env.container_id = container
    return env
