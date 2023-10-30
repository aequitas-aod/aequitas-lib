
from getpass import getpass

class Authentication():
    def __init__(self, username: str = None, password: str = None):
        if username == None and password==None:
            self.username = input("Username:")
            self.password = getpass(prompt='Password')
    
    def get_token(self) -> str:
        return ''
    
    def refresh_token(self) -> str:
        return ''