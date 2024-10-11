import time
import atexit

class TimeLogger:
    def __init__(self, log_filename, with_current_time=False):
        if with_current_time:
            current_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            self.log_filename = f"{log_filename}_{current_time_str}.txt"
        else:
            self.log_filename = log_filename

        self.logs = []
        atexit.register(self.save_logs)

    def record_moment(self, moment_name):
        current_time = time.time()
        self.logs.append((moment_name, current_time))

    def start_period(self, period_name):
        start_time = time.time()
        self.logs.append((period_name, start_time, 0))

    def end_period(self, period_name):
        end_time = time.time()
        for i in range(len(self.logs) - 1, -1, -1):
            if isinstance(self.logs[i], tuple) and len(self.logs[i]) == 3 and self.logs[i][0] == period_name:
                self.logs[i] = (self.logs[i][0], self.logs[i][1], end_time)
                break

    def save_logs(self):
        with open(self.log_filename, 'w') as file:
            for log in self.logs:
                if len(log) == 2:
                    file.write(f"{log[0]} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log[1]))}\n")
                elif len(log) == 3:
                    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log[1]))
                    assert log[2] != 0, f"End time for period '{log[0]}' has not been recorded."
                    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log[2]))
                    duration = log[2] - log[1]
                    file.write(f"{log[0]} from {start_time_str} to {end_time_str}, duration: {duration:.2f} seconds\n")
