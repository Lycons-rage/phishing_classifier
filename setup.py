from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ." #to remove this from requirements.txt

def get_requirements(file_path:str) -> List[str]:
    '''Function to get the requirements from requirements.txt and return a list having names of all the modules to be installed'''
    with open(file_path, "r") as setup_obj:
        requirement = setup_obj.readline()
        requirement = [req.replace("\n","") for req in requirement]
        
        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)
        
        return requirement


setup(
    name= "Phishing_Classifier",
    version= "0.0.1",
    author= "Chinmaya Tewari",
    author_email= "chinmayatewari.2002@gmail.com",
    install_requires= get_requirements("requirements.txt"),
    packages= find_packages()
)