NAME = 'shout'
DESCRIPTION = 'Upper-case a string.'
PARAMETERS = {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}

def run(args, workdir):
    return str(args.get('text', '')).upper()
