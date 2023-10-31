import requests
from .exceptions import UnauthorizedException

class Projects():

    @classmethod
    def list(cls, host: str = None, token: str = None):
        print(f'List projects... ')
        raise UnauthorizedException()
        pass

    @classmethod
    def get(cls, project_id: str = None, host: str = None, token: str = None):
        print(f'Getting project... {project_id}')
        raise UnauthorizedException()
        pass