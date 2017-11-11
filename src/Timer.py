import datetime


class Timer:
    def __init__(self):
        self.timer = datetime.datetime
        self.start_time = 0

    def start(self):
        self.start_time = self.timer.now()

    def stop(self):
        return (self.timer.now() - self.start_time).total_seconds()
