import functools
from .exceptions import UnauthorizedException
from .authentication import Authentication

from .projects import Projects as projects


class Aequitas_Context():
    def __init__(self, aequitas_host: str = 'localhost:9000', authentication: Authentication= None):
        self.aequitas_host = aequitas_host
        self.authentication = authentication

    def authenticated(func):
            """
            DECORATOR
            Authenticate to the minio storage service by token and refresh the token when it expires.
            """

            @functools.wraps(func)
            def context_request(self, *args, **kwargs):

                try:
                    func(self, *args, **kwargs)

                except UnauthorizedException as exc:
                    print('Refreshing token ... ')
                    self.authentication.refresh_token()
                    # func(self, *args, **kwargs)

                except Exception as exc:
                    raise exc

            return context_request

    @authenticated
    def list_project(self):
        token = self.authentication.get_token()
        return projects.list(host=self.aequitas_host, token=token)

    @authenticated
    def get_project(self, project_id: str = None):
        token = self.authentication.get_token()
        projects.get(project_id=project_id, host=self.aequitas_host, token=token)