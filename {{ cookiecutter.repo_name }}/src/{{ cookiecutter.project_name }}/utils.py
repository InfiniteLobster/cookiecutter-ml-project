#--------------------Libraries--------------------#
from pathlib import Path
#--------------------Functions--------------------#
#Function to get the project root directory
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]