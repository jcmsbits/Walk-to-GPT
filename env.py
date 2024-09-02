import os
import pprint
#var_env = os.getenv('candela')
var_ent = os.environ
print("User's Enviroment variable: ")
pprint.pprint(dict(var_ent), width=1)