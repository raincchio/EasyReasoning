import tyro
from dataclasses import dataclass

@dataclass
class Config:
    name:str = "xing"
    age:int = 19


config = tyro.cli(Config)

print(str(config))
