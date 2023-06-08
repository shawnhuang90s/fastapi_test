# -*- coding: utf-8 -*-
# @Time: 2023/6/8 10:31
import logging
import asyncio
import traceback
from abc import ABC, abstractmethod

RUNNING_FLAG = "running_flag"


class BaseTask(ABC):

    task_name: str
    sleep_seconds: int

    @abstractmethod
    async def job(self):
        raise NotImplementedError

    async def scheduled_task(self, loop) -> None:
        try:
            while getattr(loop, RUNNING_FLAG, True):
                try:
                    logging.info(f">>>>>>>>>> Task:{self.task_name} start <<<<<<<<<<")
                    await self.job()
                    logging.info(f">>>>>>>>>> Task:{self.task_name} end <<<<<<<<<<")
                except Exception as e:
                    logging.error(f"Task:{self.task_name} failed: {e}")
                    logging.error(traceback.format_exc())
                await asyncio.sleep(self.sleep_seconds)
        except asyncio.CancelledError:
            logging.info('Task has been cancelled.')


class TestTask(BaseTask):

    task_name = "TEST_TASK"
    sleep_seconds = 5

    async def job(self):
        tasks = []
        for i in range(5):
            tasks.append(self.task())
        await asyncio.gather(*tasks)

    @staticmethod
    async def task():
        print('this is a task...')


async def run(loop) -> None:
    task_obj = TestTask()
    await asyncio.gather(task_obj.scheduled_task(loop=loop))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
