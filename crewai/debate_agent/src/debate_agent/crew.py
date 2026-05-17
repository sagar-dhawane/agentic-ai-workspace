from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class DebateAgent():
    """DebateAgent crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def debater(self) -> Agent:
        """Creates the Debater/Opposer Agent"""
        return Agent(
            config=self.agents_config['debater'], # type: ignore[index]
            verbose=True
        )

    @agent
    def judge(self) -> Agent:
        """Creates the Judge Agent"""
        return Agent(
            config=self.agents_config['judge'], # type: ignore[index]
            verbose=True
        )

    @task
    def propose_task(self) -> Task:
        """Task for presenting winning arguments in favor of the topic"""
        return Task(
            config=self.tasks_config['propose_task'], # type: ignore[index]
        )

    @task
    def oppose_task(self) -> Task:
        """Task for presenting counterarguments against the topic"""
        return Task(
            config=self.tasks_config['oppose_task'], # type: ignore[index]
        )

    @task
    def judge_task(self) -> Task:
        """Task for evaluating both sides and rendering a verdict"""
        return Task(
            config=self.tasks_config['judge_task'], # type: ignore[index]
            context=[self.propose_task(), self.oppose_task()], # Feeds previous arguments to the judge
            output_file= 'output/debate_verdict.md' # Saves the final output into this file
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DebateAgent crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True
        )
