"""
CVPR-Auto Multi-Agent System
多智能体协作科研系统

架构:
- AgentOrchestrator: 协调器，负责任务分配和通信
- IdeaAgent: 想法生成智能体
- ExperimentAgent: 实验执行智能体
- WritingAgent: 论文撰写智能体
- ReviewAgent: 评审智能体
- ImprovementAgent: 改进智能体

通信机制:
- MessageBus: 消息总线，Agent间异步通信
- SharedMemory: 共享内存，存储中间结果
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import threading
import queue


class AgentRole(Enum):
    """Agent 角色类型"""
    IDEA = "idea"
    EXPERIMENT = "experiment"
    WRITING = "writing"
    REVIEW = "review"
    IMPROVEMENT = "improvement"
    ORCHESTRATOR = "orchestrator"


class MessageType(Enum):
    """消息类型"""
    TASK_ASSIGN = "task_assign"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    RESULT_SHARE = "result_share"
    REQUEST_HELP = "request_help"
    COORDINATION = "coordination"


@dataclass
class Message:
    """Agent 间消息"""
    msg_id: str
    sender: str
    receiver: str
    msg_type: MessageType
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 0  # 0-10, 越高越优先


@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: str
    description: str
    requirements: Dict[str, Any]
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class MessageBus:
    """消息总线 - Agent 间通信"""

    def __init__(self):
        self.queues: Dict[str, queue.PriorityQueue] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
        self.message_history: List[Message] = []

    def register_agent(self, agent_id: str):
        """注册 Agent"""
        with self.lock:
            if agent_id not in self.queues:
                self.queues[agent_id] = queue.PriorityQueue()
                self.subscribers[agent_id] = []

    def unregister_agent(self, agent_id: str):
        """注销 Agent"""
        with self.lock:
            self.queues.pop(agent_id, None)
            self.subscribers.pop(agent_id, None)

    def send_message(self, message: Message):
        """发送消息"""
        self.message_history.append(message)

        receiver = message.receiver
        if receiver in self.queues:
            # 优先级队列 (priority, timestamp, message)
            self.queues[receiver].put((-message.priority, time.time(), message))

        # 触发订阅回调
        if receiver in self.subscribers:
            for callback in self.subscribers[receiver]:
                try:
                    callback(message)
                except Exception as e:
                    print(f"Callback error: {e}")

    def receive_message(self, agent_id: str, timeout: float = 0.1) -> Optional[Message]:
        """接收消息"""
        if agent_id not in self.queues:
            return None

        try:
            _, _, message = self.queues[agent_id].get(timeout=timeout)
            return message
        except queue.Empty:
            return None

    def subscribe(self, agent_id: str, callback: Callable):
        """订阅消息"""
        with self.lock:
            if agent_id in self.subscribers:
                self.subscribers[agent_id].append(callback)

    def get_history(self, sender: Optional[str] = None,
                   receiver: Optional[str] = None,
                   msg_type: Optional[MessageType] = None) -> List[Message]:
        """获取消息历史"""
        messages = self.message_history
        if sender:
            messages = [m for m in messages if m.sender == sender]
        if receiver:
            messages = [m for m in messages if m.receiver == receiver]
        if msg_type:
            messages = [m for m in messages if m.msg_type == msg_type]
        return messages


class SharedMemory:
    """共享内存 - Agent 间共享数据"""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.access_log: List[Dict] = []

    def write(self, key: str, value: Any, agent_id: str):
        """写入数据"""
        with self.lock:
            self.data[key] = value
            self.access_log.append({
                "action": "write",
                "key": key,
                "agent": agent_id,
                "timestamp": datetime.now().isoformat()
            })

    def read(self, key: str, agent_id: str) -> Any:
        """读取数据"""
        with self.lock:
            value = self.data.get(key)
            self.access_log.append({
                "action": "read",
                "key": key,
                "agent": agent_id,
                "timestamp": datetime.now().isoformat()
            })
            return value

    def read_all(self, pattern: str = "", agent_id: str = "") -> Dict[str, Any]:
        """批量读取"""
        with self.lock:
            if pattern:
                return {k: v for k, v in self.data.items() if pattern in k}
            return self.data.copy()

    def delete(self, key: str, agent_id: str):
        """删除数据"""
        with self.lock:
            if key in self.data:
                del self.data[key]
                self.access_log.append({
                    "action": "delete",
                    "key": key,
                    "agent": agent_id,
                    "timestamp": datetime.now().isoformat()
                })


class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, agent_id: str, role: AgentRole,
                 message_bus: MessageBus, shared_memory: SharedMemory):
        self.agent_id = agent_id
        self.role = role
        self.message_bus = message_bus
        self.shared_memory = shared_memory
        self.is_running = False
        self.current_task: Optional[Task] = None
        self.task_history: List[Task] = []
        self.llm_client = None

        # 注册到消息总线
        self.message_bus.register_agent(self.agent_id)
        self.message_bus.subscribe(self.agent_id, self._on_message)

    def set_llm_client(self, llm_client):
        """设置 LLM 客户端"""
        self.llm_client = llm_client

    @abstractmethod
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行任务 - 子类必须实现"""
        pass

    def _on_message(self, message: Message):
        """消息回调"""
        if message.msg_type == MessageType.TASK_ASSIGN:
            self._handle_task_assign(message)

    def _handle_task_assign(self, message: Message):
        """处理任务分配"""
        task_data = message.content.get("task")
        if task_data:
            task = Task(**task_data)
            self.current_task = task
            self.run()

    def run(self):
        """运行 Agent"""
        if not self.current_task:
            return

        self.is_running = True
        task = self.current_task
        task.status = "running"
        task.assigned_to = self.agent_id

        print(f"[{self.agent_id}] 开始执行任务: {task.task_id}")

        try:
            # 执行任务
            result = self.execute_task(task)

            # 更新任务状态
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now().isoformat()

            # 发送完成消息
            self._send_completion_message(task)

            # 存储结果到共享内存
            self.shared_memory.write(
                f"result:{task.task_type}:{task.task_id}",
                result,
                self.agent_id
            )

        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            self._send_failure_message(task, str(e))

        finally:
            self.task_history.append(task)
            self.is_running = False
            self.current_task = None

    def _send_completion_message(self, task: Task):
        """发送任务完成消息"""
        message = Message(
            msg_id=f"msg_{int(time.time() * 1000)}",
            sender=self.agent_id,
            receiver="orchestrator",
            msg_type=MessageType.TASK_COMPLETE,
            content={
                "task_id": task.task_id,
                "task_type": task.task_type,
                "result": task.result,
                "agent": self.agent_id
            },
            priority=5
        )
        self.message_bus.send_message(message)

    def _send_failure_message(self, task: Task, error: str):
        """发送任务失败消息"""
        message = Message(
            msg_id=f"msg_{int(time.time() * 1000)}",
            sender=self.agent_id,
            receiver="orchestrator",
            msg_type=MessageType.TASK_FAILED,
            content={
                "task_id": task.task_id,
                "error": error,
                "agent": self.agent_id
            },
            priority=10  # 高优先级
        )
        self.message_bus.send_message(message)

    def request_help(self, help_type: str, details: Dict):
        """请求协助"""
        message = Message(
            msg_id=f"msg_{int(time.time() * 1000)}",
            sender=self.agent_id,
            receiver="orchestrator",
            msg_type=MessageType.REQUEST_HELP,
            content={
                "help_type": help_type,
                "details": details,
                "current_task": self.current_task.task_id if self.current_task else None
            }
        )
        self.message_bus.send_message(message)

    def get_status(self) -> Dict:
        """获取 Agent 状态"""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "is_running": self.is_running,
            "current_task": self.current_task.task_id if self.current_task else None,
            "task_count": len(self.task_history),
            "completed_tasks": len([t for t in self.task_history if t.status == "completed"])
        }


class AgentOrchestrator:
    """Agent 协调器 - 中央控制器"""

    def __init__(self, llm_client=None):
        self.agent_id = "orchestrator"
        self.message_bus = MessageBus()
        self.shared_memory = SharedMemory()
        self.llm_client = llm_client

        # Agent 注册表
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_roles: Dict[AgentRole, List[str]] = {role: [] for role in AgentRole}

        # 任务队列
        self.pending_tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.completed_tasks: Dict[str, Task] = {}

        # 工作流定义
        self.workflow = self._default_workflow()

        # 运行状态
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

        # 注册到消息总线
        self.message_bus.register_agent(self.agent_id)
        self.message_bus.subscribe(self.agent_id, self._on_message)

    def _default_workflow(self) -> Dict:
        """默认工作流程"""
        return {
            "stages": [
                {
                    "name": "idea_generation",
                    "agent_role": AgentRole.IDEA,
                    "input": [],
                    "output": ["ideas"],
                    "next": ["experiment"]
                },
                {
                    "name": "experiment",
                    "agent_role": AgentRole.EXPERIMENT,
                    "input": ["ideas"],
                    "output": ["experiment_results", "plots"],
                    "next": ["writing"]
                },
                {
                    "name": "writing",
                    "agent_role": AgentRole.WRITING,
                    "input": ["experiment_results", "plots"],
                    "output": ["paper_draft"],
                    "next": ["review"]
                },
                {
                    "name": "review",
                    "agent_role": AgentRole.REVIEW,
                    "input": ["paper_draft"],
                    "output": ["review_report"],
                    "next": ["improvement", "finalize"]
                },
                {
                    "name": "improvement",
                    "agent_role": AgentRole.IMPROVEMENT,
                    "input": ["paper_draft", "review_report"],
                    "output": ["improved_paper"],
                    "next": ["review"]
                },
                {
                    "name": "finalize",
                    "agent_role": None,
                    "input": ["review_report"],
                    "output": ["final_paper"],
                    "next": []
                }
            ]
        }

    def register_agent(self, agent: BaseAgent):
        """注册 Agent"""
        agent.set_llm_client(self.llm_client)
        self.agents[agent.agent_id] = agent
        self.agent_roles[agent.role].append(agent.agent_id)
        print(f"[Orchestrator] 注册 Agent: {agent.agent_id} ({agent.role.value})")

    def unregister_agent(self, agent_id: str):
        """注销 Agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            self.agent_roles[agent.role].remove(agent_id)
            del self.agents[agent_id]
            self.message_bus.unregister_agent(agent_id)

    def submit_task(self, task: Task, priority: int = 0):
        """提交任务"""
        self.pending_tasks.put((-priority, time.time(), task))
        print(f"[Orchestrator] 提交任务: {task.task_id} (优先级: {priority})")

    def _on_message(self, message: Message):
        """处理消息"""
        if message.msg_type == MessageType.TASK_COMPLETE:
            self._handle_task_complete(message)
        elif message.msg_type == MessageType.TASK_FAILED:
            self._handle_task_failed(message)
        elif message.msg_type == MessageType.REQUEST_HELP:
            self._handle_help_request(message)

    def _handle_task_complete(self, message: Message):
        """处理任务完成"""
        content = message.content
        task_id = content["task_id"]
        result = content["result"]
        agent_id = content["agent"]

        print(f"[Orchestrator] 任务完成: {task_id} by {agent_id}")

        # 存储结果
        self.completed_tasks[task_id] = result

        # 触发工作流下一步
        self._trigger_next_stage(task_id, result)

    def _handle_task_failed(self, message: Message):
        """处理任务失败"""
        content = message.content
        task_id = content["task_id"]
        error = content["error"]

        print(f"[Orchestrator] 任务失败: {task_id}, 错误: {error}")

        # 可以在这里实现重试逻辑
        # 或者分配给另一个 Agent

    def _handle_help_request(self, message: Message):
        """处理帮助请求"""
        content = message.content
        help_type = content["help_type"]
        details = content["details"]

        print(f"[Orchestrator] 帮助请求: {help_type}")

        # 根据帮助类型分配给合适的 Agent
        if help_type == "llm_query":
            # 直接响应 LLM 查询
            pass
        elif help_type == "tool_use":
            # 调用工具
            pass

    def _trigger_next_stage(self, completed_task_id: str, result: Dict):
        """触发工作流下一阶段"""
        # 查找当前阶段
        for stage in self.workflow["stages"]:
            if completed_task_id.startswith(stage["name"]):
                next_stages = stage.get("next", [])
                for next_stage_name in next_stages:
                    self._create_stage_task(next_stage_name, result)
                break

    def _create_stage_task(self, stage_name: str, input_data: Dict):
        """创建阶段任务"""
        # 查找阶段定义
        stage_def = None
        for stage in self.workflow["stages"]:
            if stage["name"] == stage_name:
                stage_def = stage
                break

        if not stage_def:
            return

        # 创建任务
        task = Task(
            task_id=f"{stage_name}_{int(time.time())}",
            task_type=stage_name,
            description=f"Execute {stage_name} stage",
            requirements={
                "input": stage_def["input"],
                "output": stage_def["output"],
                "input_data": input_data
            }
        )

        # 分配给合适的 Agent
        agent_role = stage_def["agent_role"]
        if agent_role:
            available_agents = self.agent_roles.get(agent_role, [])
            if available_agents:
                # 选择负载最低的 Agent
                agent_id = self._select_best_agent(available_agents)
                task.assigned_to = agent_id

                # 发送任务分配消息
                message = Message(
                    msg_id=f"msg_{int(time.time() * 1000)}",
                    sender=self.agent_id,
                    receiver=agent_id,
                    msg_type=MessageType.TASK_ASSIGN,
                    content={"task": task.__dict__},
                    priority=5
                )
                self.message_bus.send_message(message)

    def _select_best_agent(self, agent_ids: List[str]) -> str:
        """选择最佳 Agent（负载均衡）"""
        best_agent = agent_ids[0]
        min_load = float('inf')

        for agent_id in agent_ids:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                status = agent.get_status()
                load = 1 if status["is_running"] else 0
                if load < min_load:
                    min_load = load
                    best_agent = agent_id

        return best_agent

    def start(self):
        """启动 Orchestrator"""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()
        print("[Orchestrator] 已启动")

    def stop(self):
        """停止 Orchestrator"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()
        print("[Orchestrator] 已停止")

    def _worker_loop(self):
        """工作线程"""
        while self.is_running:
            # 处理消息
            message = self.message_bus.receive_message(self.agent_id, timeout=0.1)
            if message:
                self._on_message(message)

            time.sleep(0.1)

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "orchestrator": {
                "is_running": self.is_running,
                "pending_tasks": self.pending_tasks.qsize(),
                "completed_tasks": len(self.completed_tasks)
            },
            "agents": {
                role.value: [self.agents[aid].get_status() for aid in agent_ids]
                for role, agent_ids in self.agent_roles.items()
            },
            "message_bus": {
                "history_count": len(self.message_bus.message_history)
            },
            "shared_memory": {
                "keys_count": len(self.shared_memory.data)
            }
        }

    def run_research_project(self, config: Dict) -> Dict:
        """运行完整科研项目"""
        print("=" * 70)
        print("🚀 启动多智能体科研项目")
        print("=" * 70)

        # 创建初始任务
        initial_task = Task(
            task_id=f"idea_generation_{int(time.time())}",
            task_type="idea_generation",
            description="Generate novel research ideas",
            requirements={
                "domain": config.get("domain", "computer_vision"),
                "num_ideas": config.get("num_ideas", 5),
                "constraints": config.get("constraints", {})
            }
        )

        self.submit_task(initial_task, priority=10)

        # 等待所有任务完成
        # 简化版本：实际应该等待工作流完成
        return {
            "status": "started",
            "initial_task": initial_task.task_id,
            "config": config
        }


if __name__ == "__main__":
    # 测试基础组件
    print("测试 Multi-Agent System...")

    # 创建组件
    bus = MessageBus()
    memory = SharedMemory()
    orchestrator = AgentOrchestrator()

    print("✓ 基础组件创建成功")
    print(f"  - MessageBus 注册")
    print(f"  - SharedMemory 注册")
    print(f"  - Orchestrator 注册")
