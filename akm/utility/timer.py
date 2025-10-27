import json


class Timer:
    def __init__(self):
        self.time_cost = {}
    
    def update(self, func_name, duration):
        if func_name not in self.time_cost:
            self.time_cost[func_name] = []
        self.time_cost[func_name].append(duration)
            
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.time_cost, f, ensure_ascii=False, indent=2)
