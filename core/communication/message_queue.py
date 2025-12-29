import threading
from collections import deque
import copy

class MessageQueue(deque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def append(self, x):
        with self.lock:
            super().append(x)

    def appendleft(self, x):
        with self.lock:
            super().appendleft(x)

    def pop(self):
        with self.lock:
            return super().pop()

    def popleft(self):
        with self.lock:
            return super().popleft()

    def copy(self):
        with self.lock:
            new_deque = MessageQueue(copy.deepcopy(list(self)))
            return new_deque
        
    def size(self):
        with self.lock:
            return len(self)
        
    def empty(self):
        with self.lock:
            return len(self) == 0
