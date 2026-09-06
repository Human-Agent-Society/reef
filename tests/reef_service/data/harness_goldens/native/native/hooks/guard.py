def listen(payload, next):
    return next()

NAME = 'guard'
EVENT = 'post_execute'
