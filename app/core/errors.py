"""跨模块共享的异常类型。"""


class TaskCancelled(RuntimeError):
    """用户主动取消任务时抛出；Pipeline 捕获后向 GUI 上报正常终止。"""
