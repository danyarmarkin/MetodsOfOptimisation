class Func:
    def __init__(self, func):
        self.func = func
        self.counter_ = 0
        self.grad_counter_ = 0

    def __call__(self, x):
        self.counter_ += 1
        return self.func(*x)

    def grad(self):
        self.grad_counter_ += 1

    @property
    def calls(self):
        return self.counter_

    @property
    def grads(self):
        return self.grad_counter_