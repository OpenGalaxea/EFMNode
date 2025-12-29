import toml
from scheduler.scheduler import Scheduler

def main():
    config = toml.load("config.toml")
    scheduler = Scheduler(config)
    scheduler.run()

if __name__ == "__main__":
    main()